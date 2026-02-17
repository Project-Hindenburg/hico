"""
Shared visualisation helpers for embedding analysis notebooks.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import umap


# ── I/O helpers ───────────────────────────────────────────────────────


def load_token_id_map(path: Path, encoding="utf-8"):
    """
    Read a two-column file (token  id) and return (token_to_id, id_to_token).
    Robust parsing: takes the last whitespace-separated field as the id.
    """
    token_to_id = {}
    id_to_token = {}

    with path.open("r", encoding=encoding) as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                token, id_str = line.rsplit(None, 1)
            except ValueError:
                raise ValueError(f"Line {ln} does not have 2 columns: {line!r}")
            try:
                tid = int(id_str)
            except ValueError:
                raise ValueError(f"Line {ln} id not integer: {id_str!r}")
            token_to_id[token] = tid
            id_to_token[tid] = token

    return token_to_id, id_to_token


def load_grid_structure(path: Path, token_to_id: dict, encoding="utf-8"):
    """
    Read a 4x4 whitespace-separated grid.
    Returns: GRID, WORD_TO_POS, WORD_TO_TID, ID_TO_WORD, ROW_LABELS
    """
    rows = []
    with path.open("r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split())

    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        raise ValueError(
            f"Expected 4x4 grid, got {len(rows)} rows "
            f"with lengths {[len(r) for r in rows]}"
        )

    GRID = rows
    WORD_TO_POS = {w: (r, c) for r, row in enumerate(GRID) for c, w in enumerate(row)}

    missing = [w for w in WORD_TO_POS if w not in token_to_id]
    if missing:
        raise KeyError(
            "Grid tokens missing from token-id file: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )

    WORD_TO_TID = {w: token_to_id[w] for w in WORD_TO_POS}
    ID_TO_WORD = {tid: w for w, tid in WORD_TO_TID.items()}
    ROW_LABELS = [f"Row {r}: " + " ".join(GRID[r]) for r in range(4)]

    return GRID, WORD_TO_POS, WORD_TO_TID, ID_TO_WORD, ROW_LABELS


def load_embeddings_pt(pt_path: Path, emb_key="embeddings_last", ids_key="input_ids_last"):
    """
    Load a .pt file and return (ids_np, emb_np).
    Handles optional (1, L) / (1, L, D) batch dimensions.
    """
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    ids = obj[ids_key]
    emb = obj[emb_key]

    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb[0]

    return ids.detach().cpu().numpy(), emb.detach().cpu().float().numpy()


# ── Tree structure parsers ────────────────────────────────────────────


def load_tree_structure(path: Path, token_to_id: dict, encoding="utf-8"):
    """
    Parse a tree structure.
    Returns: tokens, token_to_depth, edges, WORD_TO_TID, ID_TO_WORD, depth_labels
    """
    lines = path.read_text(encoding=encoding).strip().splitlines()
    tokens = []
    token_to_depth = {}
    parent_at_depth = {}
    edges_words = []

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue

        pos = None
        for j, ch in enumerate(stripped):
            if ch in ("├", "└"):
                pos = j
                break

        if pos is None:
            token = stripped.strip()
            depth = 0
        else:
            depth = pos // 4 + 1
            token = stripped[pos:].lstrip("├└─ ").strip()

        tokens.append(token)
        token_to_depth[token] = depth
        parent_at_depth[depth] = token
        if depth > 0 and (depth - 1) in parent_at_depth:
            edges_words.append((parent_at_depth[depth - 1], token))

    missing = [t for t in tokens if t not in token_to_id]
    if missing:
        raise KeyError(f"Missing tokens: {missing[:8]}")

    WORD_TO_TID = {t: token_to_id[t] for t in tokens}
    ID_TO_WORD = {tid: t for t, tid in WORD_TO_TID.items()}

    edges = [
        (min(WORD_TO_TID[p], WORD_TO_TID[c]), max(WORD_TO_TID[p], WORD_TO_TID[c]))
        for p, c in edges_words
    ]

    max_d = max(token_to_depth.values())
    depth_labels = {d: f"Depth {d}" for d in range(max_d + 1)}

    return tokens, token_to_depth, edges, WORD_TO_TID, ID_TO_WORD, depth_labels


def load_tree_cluster_structure(path: Path, token_to_id: dict, encoding="utf-8"):
    """
    Parse a cluster tree where each node = (tok1, tok2, tok3).
    Returns: clusters, token_to_depth, inter_edges, intra_edges,
             WORD_TO_TID, ID_TO_WORD, depth_labels
    """
    lines = path.read_text(encoding=encoding).strip().splitlines()
    clusters = []
    cluster_depths = []
    parent_cluster_at_depth = {}
    token_to_depth = {}
    tree_links = []

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        try:
            s, e = stripped.index("("), stripped.index(")")
        except ValueError:
            continue

        cluster_tokens = [t.strip() for t in stripped[s + 1 : e].split(",")]

        pos = None
        for j, ch in enumerate(stripped):
            if ch in ("├", "└"):
                pos = j
                break
        depth = 0 if pos is None else pos // 4 + 1

        idx = len(clusters)
        clusters.append(cluster_tokens)
        cluster_depths.append(depth)
        for t in cluster_tokens:
            token_to_depth[t] = depth
        parent_cluster_at_depth[depth] = idx
        if depth > 0 and (depth - 1) in parent_cluster_at_depth:
            tree_links.append((parent_cluster_at_depth[depth - 1], idx))

    all_tokens = [t for cl in clusters for t in cl]
    missing = [t for t in all_tokens if t not in token_to_id]
    if missing:
        raise KeyError(f"Missing tokens: {missing[:8]}")

    WORD_TO_TID = {t: token_to_id[t] for t in all_tokens}
    ID_TO_WORD = {tid: t for t, tid in WORD_TO_TID.items()}

    inter_edges = []
    for pi, ci in tree_links:
        for pt in clusters[pi]:
            for ct in clusters[ci]:
                t1, t2 = WORD_TO_TID[pt], WORD_TO_TID[ct]
                inter_edges.append((min(t1, t2), max(t1, t2)))

    intra_edges = []
    for cl in clusters:
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                t1, t2 = WORD_TO_TID[cl[a]], WORD_TO_TID[cl[b]]
                intra_edges.append((min(t1, t2), max(t1, t2)))

    max_d = max(token_to_depth.values())
    depth_labels = {d: f"Depth {d}" for d in range(max_d + 1)}

    return (
        clusters, token_to_depth, inter_edges, intra_edges,
        WORD_TO_TID, ID_TO_WORD, depth_labels,
    )


# ── Edge builders ─────────────────────────────────────────────────────


def grid_edges_from_grid(GRID, WORD_TO_TID, available_tids=None):
    """
    Return (tid1, tid2) pairs for 4-connected adjacent cells in GRID.
    Optionally filter to pairs where both tids are in *available_tids*.
    """
    n_rows = len(GRID)
    n_cols = len(GRID[0]) if n_rows else 0
    edges = []
    for r in range(n_rows):
        for c in range(n_cols):
            t1 = WORD_TO_TID[GRID[r][c]]
            for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    t2 = WORD_TO_TID[GRID[rr][cc]]
                    if t1 < t2:
                        if available_tids is None or (t1 in available_tids and t2 in available_tids):
                            edges.append((t1, t2))
    return edges


def torus_edges_from_grid(GRID, WORD_TO_TID):
    """Like grid_edges_from_grid but with top↔bottom and left↔right wrap."""
    n_rows, n_cols = len(GRID), len(GRID[0])
    edges = set()
    for r in range(n_rows):
        for c in range(n_cols):
            t1 = WORD_TO_TID[GRID[r][c]]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = (r + dr) % n_rows, (c + dc) % n_cols
                t2 = WORD_TO_TID[GRID[rr][cc]]
                if t1 != t2:
                    edges.add((min(t1, t2), max(t1, t2)))
    return list(edges)


# ── Internal windowing helper ─────────────────────────────────────────

# TODO: WHY ARE THERE UNKNOWN TOKEN IDs?
def _window_and_filter(all_ids_np, all_emb_np, ID_TO_WORD, NW, title=""):
    """Take the last NW tokens and drop those not in ID_TO_WORD."""
    L = len(all_ids_np)
    W = min(NW, L)
    win_ids = all_ids_np[-W:]
    win_emb = all_emb_np[-W:]

    known_tids = set(ID_TO_WORD.keys())
    keep_mask = np.isin(win_ids, list(known_tids))
    n_discard = int((~keep_mask).sum())
    if n_discard > 0:
        print(f"  [{title}] Discarded {n_discard}/{len(win_ids)} embeddings with unknown token IDs")
    return win_ids[keep_mask], win_emb[keep_mask]


# ── Plotting functions ────────────────────────────────────────────────


def plot_pca_on_ax(
    ax,
    all_ids_np,
    all_emb_np,
    ID_TO_WORD,
    token_to_group,
    group_colors,
    group_labels,
    edges,
    NW=500,
    random_state=0,
    point_size=40,
    alpha=0.7,
    title="",
    annotate=True,
    font_size=7,
):
    """PCA 2D scatter + edges on a given Axes."""
    win_ids, win_emb = _window_and_filter(all_ids_np, all_emb_np, ID_TO_WORD, NW, title)

    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(win_emb)

    unique_tids = np.unique(win_ids)
    centroids = {int(t): Z[win_ids == t].mean(axis=0) for t in unique_tids}

    avail = set(centroids)
    for t1, t2 in edges:
        if t1 in avail and t2 in avail:
            c1, c2 = centroids[t1], centroids[t2]
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color="#cccccc", lw=0.8, zorder=1)

    legend_added = set()
    for tid in unique_tids:
        tid_int = int(tid)
        mask = win_ids == tid
        word = ID_TO_WORD[tid_int]
        grp = token_to_group.get(word)
        color = group_colors.get(grp, "grey")
        label = group_labels.get(grp) if grp not in legend_added else None
        if grp is not None:
            legend_added.add(grp)

        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=point_size, alpha=alpha, color=color,
            edgecolors="k", linewidths=0.3, label=label, zorder=3,
        )
        if annotate:
            cx, cy = centroids[tid_int]
            ax.annotate(
                word, (cx, cy), textcoords="offset points",
                xytext=(6, 5), fontsize=font_size, fontweight="bold",
            )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.set_title(f"{title}\nVar: {ev[0]:.1%} + {ev[1]:.1%}", fontsize=9)
    ax.legend(fontsize=6, framealpha=0.7, loc="best")
    ax.grid(True, alpha=0.15)


def plot_pca_3d_on_ax(
    ax,
    all_ids_np,
    all_emb_np,
    ID_TO_WORD,
    token_to_group,
    group_colors,
    group_labels,
    edges,
    NW=500,
    random_state=0,
    point_size=40,
    alpha=0.7,
    title="",
    annotate=True,
    font_size=7,
):
    """PCA 3D scatter + edges on a given 3D Axes."""
    win_ids, win_emb = _window_and_filter(all_ids_np, all_emb_np, ID_TO_WORD, NW, title)

    pca = PCA(n_components=3, random_state=random_state)
    Z = pca.fit_transform(win_emb)

    unique_tids = np.unique(win_ids)
    centroids = {int(t): Z[win_ids == t].mean(axis=0) for t in unique_tids}

    avail = set(centroids)
    for t1, t2 in edges:
        if t1 in avail and t2 in avail:
            c1, c2 = centroids[t1], centroids[t2]
            ax.plot(
                [c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                color="#cccccc", lw=0.8, zorder=1,
            )

    legend_added = set()
    for tid in unique_tids:
        tid_int = int(tid)
        mask = win_ids == tid
        word = ID_TO_WORD[tid_int]
        grp = token_to_group.get(word)
        color = group_colors.get(grp, "grey")
        label = group_labels.get(grp) if grp not in legend_added else None
        if grp is not None:
            legend_added.add(grp)

        ax.scatter(
            Z[mask, 0], Z[mask, 1], Z[mask, 2],
            s=point_size, alpha=alpha, color=color,
            edgecolors="k", linewidths=0.3, label=label, zorder=3,
        )
        if annotate:
            cx, cy, cz = centroids[tid_int]
            ax.text(cx, cy, cz, f"  {word}", fontsize=font_size, fontweight="bold", zorder=4)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.set_zlabel("PC 3", fontsize=8)
    ax.set_title(f"{title}\nVar: {ev[0]:.1%} + {ev[1]:.1%} + {ev[2]:.1%}", fontsize=9)
    ax.legend(fontsize=6, framealpha=0.7, loc="best")


def plot_dendrogram_on_ax(
    ax,
    all_ids_np,
    all_emb_np,
    ID_TO_WORD,
    token_to_depth,
    depth_colors,
    NW=500,
    title="",
    leaf_font_size=7,
    method="ward",
):
    """
    Per-token centroids -> Ward linkage -> horizontal dendrogram.
    Leaves coloured by true tree depth.
    """
    win_ids, win_emb = _window_and_filter(all_ids_np, all_emb_np, ID_TO_WORD, NW, title)

    unique_tids = np.unique(win_ids)
    labels = []
    centroids = []
    leaf_colors = []
    leaf_depths = []
    for tid in unique_tids:
        tid_int = int(tid)
        word = ID_TO_WORD[tid_int]
        labels.append(word)
        centroids.append(win_emb[win_ids == tid].mean(axis=0))
        depth = token_to_depth.get(word, 0)
        leaf_depths.append(depth)
        leaf_colors.append(depth_colors.get(depth, "grey"))

    centroids = np.stack(centroids)
    Z = linkage(centroids, method=method)
    leaf_color_map = {lab: col for lab, col in zip(labels, leaf_colors)}

    dendrogram(
        Z, labels=labels, ax=ax, orientation="left",
        leaf_font_size=leaf_font_size, link_color_func=lambda k: "#888888",
    )

    for lbl in ax.get_yticklabels():
        txt = lbl.get_text()
        lbl.set_color(leaf_color_map.get(txt, "black"))
        lbl.set_fontweight("bold")

    # Add legend for depths
    import matplotlib.patches as mpatches
    # Collect all depths and their colors present in this plot.
    depth_to_color = {}
    for depth, color in zip(leaf_depths, leaf_colors):
        depth_to_color[depth] = color
    handles = [
        mpatches.Patch(color=c, label=f"Depth {d}") for d, c in sorted(depth_to_color.items())
    ]
    if handles:
        ax.legend(handles=handles, title="Depth", fontsize=6, title_fontsize=7, loc="upper left", framealpha=0.7)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Ward distance", fontsize=9)
    ax.tick_params(axis="y", labelsize=leaf_font_size)


def plot_umap_on_ax(
    ax,
    all_ids_np,
    all_emb_np,
    ID_TO_WORD,
    token_to_group,
    group_colors,
    group_labels,
    edges,
    NW=500,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=0,
    point_size=40,
    alpha=0.7,
    title="",
    annotate=True,
    font_size=7,
):
    """
    Per-token centroids → UMAP (2D or 3D) → scatter coloured by group.
    Pass n_components=3 and a 3D axes for 3D plots.
    """
    win_ids, win_emb = _window_and_filter(all_ids_np, all_emb_np, ID_TO_WORD, NW, title)

    unique_tids = np.unique(win_ids)
    labels_list = []
    centroids_list = []
    for tid in unique_tids:
        tid_int = int(tid)
        labels_list.append(ID_TO_WORD[tid_int])
        centroids_list.append(win_emb[win_ids == tid].mean(axis=0))

    centroids_arr = np.stack(centroids_list)
    n_nb = min(n_neighbors, len(centroids_arr) - 1)
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_nb,
        min_dist=min_dist, random_state=random_state,
    )
    Z = reducer.fit_transform(centroids_arr)

    is_3d = n_components >= 3
    tid_to_idx = {int(tid): i for i, tid in enumerate(unique_tids)}
    avail = set(tid_to_idx)
    for t1, t2 in edges:
        if t1 in avail and t2 in avail:
            i1, i2 = tid_to_idx[t1], tid_to_idx[t2]
            if is_3d:
                ax.plot(
                    [Z[i1, 0], Z[i2, 0]], [Z[i1, 1], Z[i2, 1]], [Z[i1, 2], Z[i2, 2]],
                    color="#cccccc", lw=0.8, zorder=1,
                )
            else:
                ax.plot(
                    [Z[i1, 0], Z[i2, 0]], [Z[i1, 1], Z[i2, 1]],
                    color="#cccccc", lw=0.8, zorder=1,
                )

    legend_added = set()
    for i, word in enumerate(labels_list):
        grp = token_to_group.get(word)
        color = group_colors.get(grp, "grey")
        label = group_labels.get(grp) if grp not in legend_added else None
        if grp is not None:
            legend_added.add(grp)

        if is_3d:
            ax.scatter(
                Z[i, 0], Z[i, 1], Z[i, 2],
                s=point_size, alpha=alpha, color=color,
                edgecolors="k", linewidths=0.3, label=label, zorder=3,
            )
            if annotate:
                ax.text(
                    Z[i, 0], Z[i, 1], Z[i, 2], f"  {word}",
                    fontsize=font_size, fontweight="bold", zorder=4,
                )
        else:
            ax.scatter(
                Z[i, 0], Z[i, 1],
                s=point_size, alpha=alpha, color=color,
                edgecolors="k", linewidths=0.3, label=label, zorder=3,
            )
            if annotate:
                ax.annotate(
                    word, (Z[i, 0], Z[i, 1]), textcoords="offset points",
                    xytext=(6, 5), fontsize=font_size, fontweight="bold",
                )

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    if is_3d:
        ax.set_zlabel("UMAP 3", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, framealpha=0.7, loc="best")
    if not is_3d:
        ax.grid(True, alpha=0.15)
