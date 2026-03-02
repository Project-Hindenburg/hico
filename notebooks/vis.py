"""
Shared visualisation helpers for embedding analysis notebooks.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram


class Plotter:
    def __init__(self, vocab_path, structure_path, emb_dir):
            self.token_to_id, self.id_to_token = load_token_id_map(vocab_path)
            self.structure_path = structure_path
            self.emb_dir = emb_dir
            if "grid" in str(structure_path).lower() or "torus" in str(structure_path).lower():
                self.structure, self.word_to_pos, self.word_to_tid, self.id_to_word, self.row_labels = load_grid_structure(structure_path, self.token_to_id)

                if "torus" in str(structure_path).lower():
                    self.all_edges = torus_edges_from_grid(self.structure, self.word_to_tid)
                else:
                    self.all_edges = grid_edges_from_grid(self.structure, self.word_to_tid)

                n_rows = len(self.structure)
                base_colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261', '#8d99ae']
                ROW_COLORS = [base_colors[i % len(base_colors)] for i in range(n_rows)]

                self.row_colors_map = {i: c for i, c in enumerate(ROW_COLORS)}
                self.row_labels_map = {r: f"Row {r}: " + " ".join(self.structure[r]) for r in range(n_rows)}
                self.token_to_group = {
                    w: r for r, row in enumerate(self.structure)
                    for c, w in enumerate(row)
                }

            elif "tree" in str(structure_path).lower():
                self.tokens, self.token_to_group, self.all_edges, self.word_to_tid, self.id_to_word, self.row_labels_map = load_tree_structure(structure_path, self.token_to_id)
                self.row_colors_map = {i: c for i, c in enumerate(['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261'])}

            elif "tree_cluster_structure" in str(structure_path).lower():
                self.tokens, self.token_to_group, self.inter_edges, self.intra_edges, self.word_to_tid, self.id_to_word, self.row_labels_map = load_tree_cluster_structure(structure_path, self.token_to_id)
                self.all_edges = self.inter_edges + self.intra_edges
                self.row_colors_map = {i: c for i, c in enumerate(['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261'])}
            
            elif "circle" in str(structure_path).lower():
                (
                    self.tokens,
                    self.token_to_group,
                    self.all_edges,
                    self.word_to_tid,
                    self.id_to_word,
                    self.row_labels_map,
                ) = load_circle_structure(structure_path, self.token_to_id)
                
                n = len(self.tokens)

                if n <= 10:
                    # ---- qualitative colormap (distinct colors) ----
                    cmap = plt.cm.get_cmap("tab10", n)
                    colors = [cmap(i) for i in range(n)]

                    self.row_colors_map = {
                        i: colors[i] for i in range(n)
                    }
                else:
                    # ---- gradient colouring ----
                    cmap = plt.cm.viridis
                    colors = cmap(np.linspace(0.4, 0.95, n))

                    self.row_colors_map = {
                        i: colors[i] for i in range(n)
                    }

            
    def plot_reduced_emb(
        self, ax, reduced_emb, win_ids, title="", annotate=True, only_centroid=False, second_components=False,  add_legend=False, pc1=0, pc2=1
    ):
        point_size = 40
        centroid_size = 180   # bigger marker for centroids
        alpha = 0.7
        font_size = 7

        n_components = reduced_emb.shape[1]
        is_3d = n_components == 3


        unique_tids = np.unique(win_ids)
        centroids = {
            int(t): reduced_emb[win_ids == t].mean(axis=0)
            for t in unique_tids
        }

        # ---- Draw edges between centroids ----
        avail = set(centroids)
        for t1, t2 in self.all_edges:
            if t1 in avail and t2 in avail:
                c1, c2 = centroids[t1], centroids[t2]
                if is_3d:
                    ax.plot(
                        [c1[0], c2[0]],
                        [c1[1], c2[1]],
                        [c1[2], c2[2]],
                        color="#cccccc", lw=0.8, zorder=1
                    )
                else:
                    ax.plot(
                        [c1[0], c2[0]],
                        [c1[1], c2[1]],
                        color="#cccccc", lw=0.8, zorder=1
                    )

        legend_added = set()

        for tid in unique_tids:
            tid_int = int(tid)
            word = self.id_to_word[tid_int]

            grp = self.token_to_group.get(word)
            color = self.row_colors_map.get(grp, "grey")
            label = self.row_labels_map.get(grp) if grp not in legend_added else None
            if grp is not None:
                legend_added.add(grp)

            c = centroids[tid_int]

            # ---- Plot original points (unless only_centroid=True) ----
            if not only_centroid:
                mask = win_ids == tid

                if is_3d:
                    ax.scatter(
                        reduced_emb[mask, 0],
                        reduced_emb[mask, 1],
                        reduced_emb[mask, 2],
                        s=point_size, alpha=alpha,
                        color=color, edgecolors="k",
                        linewidths=0.3, zorder=2
                    )
                else:
                    ax.scatter(
                        reduced_emb[mask, 0],
                        reduced_emb[mask, 1],
                        s=point_size, alpha=alpha,
                        color=color, edgecolors="k",
                        linewidths=0.3, zorder=2
                    )

            # ---- Plot centroid (always plotted) ----
            if is_3d:
                ax.scatter(
                    c[0], c[1], c[2],
                    s=centroid_size,
                    color=color,
                    edgecolors="black",
                    linewidths=1.2,
                    marker="X",
                    label=label,
                    zorder=4
                )
            else:
                ax.scatter(
                    c[0], c[1],
                    s=centroid_size,
                    color=color,
                    edgecolors="black",
                    linewidths=1.2,
                    marker="X",
                    label=label,
                    zorder=4
                )

            # ---- Annotation (at centroid) ----
            if annotate:
                if is_3d:
                    ax.text(
                        c[0], c[1], c[2],
                        word,
                        fontsize=font_size,
                        fontweight="bold"
                    )
                else:
                    ax.annotate(
                        word, (c[0], c[1]),
                        textcoords="offset points",
                        xytext=(6, 5),
                        fontsize=font_size,
                        fontweight="bold"
                    )

        # ---- Labels ----
        ax.set_xlabel(f"Component {pc1+1}", fontsize=15)
        ax.set_ylabel(f"Component {pc2+1}", fontsize=15)
        if is_3d:
            ax.set_zlabel("Component 3", fontsize=15)

        ax.set_title(title, fontsize=15)
        if add_legend == True:
            ax.legend(fontsize=15, framealpha=0.7, loc="best")
        ax.grid(True, alpha=0.15)

    def plot_dendrogram(
        self,
        ax,
        all_ids_np,
        all_emb_np,
        title="",
        method="ward",
    ):
        """
        Per-token centroids -> Ward linkage -> horizontal dendrogram.
        Leaves coloured by true tree depth.
        """

        unique_tids = np.unique(all_ids_np)
        labels = []
        centroids = []
        leaf_colors = []
        leaf_depths = []
        for tid in unique_tids:
            tid_int = int(tid)
            word = self.id_to_word[tid_int]
            labels.append(word)
            centroids.append(all_emb_np[all_ids_np == tid].mean(axis=0))
            depth = self.token_to_group.get(word, 0)
            leaf_depths.append(depth)
            leaf_colors.append(self.row_colors_map.get(depth, "grey"))

        centroids = np.stack(centroids)
        Z = linkage(centroids, method=method)
        leaf_color_map = {lab: col for lab, col in zip(labels, leaf_colors)}

        dendrogram(
            Z, labels=labels, ax=ax, orientation="left", p=40,
            leaf_font_size=7, link_color_func=lambda k: "#888888",
        )

        for lbl in ax.get_yticklabels():
            txt = lbl.get_text()
            lbl.set_color(leaf_color_map.get(txt, "black"))
            lbl.set_fontweight("bold")

        # Add legend for depths
        # Collect all depths and their colors present in this plot.
        depth_to_color = {}
        for depth, color in zip(leaf_depths, leaf_colors):
            depth_to_color[depth] = color
        handles = [
            mpatches.Patch(color=c, label=f"Depth {d}") for d, c in sorted(depth_to_color.items())
        ]
        if handles:
            ax.legend(handles=handles, title="Depth", fontsize=6, title_fontsize=7, loc="upper left", framealpha=0.7)

        ax.set_title(title, fontsize=15)
        ax.set_xlabel(f"{method} distance", fontsize=15)
        ax.tick_params(axis="y", labelsize=7)

    def plot_dendrogram_on_points(
            self,
            ax,
            all_ids_np,
            all_emb_np,
            title="",
            method="ward",
            orientation="left",
            leaf_font_size=17,
            max_leaves=40,
            label_mode="token",   # "token" oppure "token+idx"
        ):
        """
        all_emb_np (N, D) -> linkage -> dendrogram.
        Leaves coloured by true tree depth (derived from token id per row).
        """

        # Safety: ensure 2D
        all_emb_np = np.asarray(all_emb_np)
        all_ids_np = np.asarray(all_ids_np)
        assert all_emb_np.ndim == 2, "all_emb_np must be 2D (N, D)"
        assert all_ids_np.shape[0] == all_emb_np.shape[0], "all_ids_np and all_emb_np must have same length"

        # Labels and colors PER POINT (not per unique token)
        labels = []
        leaf_depths = []
        leaf_colors = []

        for i, tid in enumerate(all_ids_np):
            tid_int = int(tid)
            word = self.id_to_word.get(tid_int, str(tid_int))

            if label_mode == "token+idx":
                lab = f"{word}#{i}"
            else:
                lab = word  # may repeat; fine, but ambiguous visually

            labels.append(lab)

            depth = self.token_to_group.get(word, 0)
            leaf_depths.append(depth)
            leaf_colors.append(self.row_colors_map.get(depth, "grey"))

        # Linkage directly on points
        Z = linkage(all_emb_np, method=method)

        # Map label -> color for tick labels
        # If labels repeat and label_mode=="token", this will color all same-token labels equally (OK).
        # If you want per-point uniqueness, use label_mode=="token+idx".
        leaf_color_map = {lab: col for lab, col in zip(labels, leaf_colors)}

        dendrogram(
            Z,
            labels=labels,
            ax=ax,
            orientation=orientation,
            p=max_leaves,
            leaf_font_size=leaf_font_size,
            link_color_func=lambda k: "#888888",
            truncate_mode="lastp" if max_leaves is not None else None,
            show_leaf_counts=True if max_leaves is not None else False,
        )

        # Color y tick labels (orientation="left" => leaves on y-axis)
        ticklabels = ax.get_yticklabels() if orientation in ("left", "right") else ax.get_xticklabels()
        for lbl in ticklabels:
            txt = lbl.get_text()
            lbl.set_color(leaf_color_map.get(txt, "black"))
            lbl.set_fontweight("bold")

        # Legend for depths present
        depth_to_color = {}
        for d, c in zip(leaf_depths, leaf_colors):
            depth_to_color[d] = c
        handles = [mpatches.Patch(color=c, label=f"Depth {d}") for d, c in sorted(depth_to_color.items())]
        if handles:
            ax.legend(handles=handles, title="Depth", fontsize=12, title_fontsize=12,
                    loc="upper left", framealpha=0.7)

        ax.set_title(title, fontsize=17)
        ax.set_xlabel(f"{method} distance", fontsize=17)
        ax.tick_params(axis="y" if orientation in ("left", "right") else "x", labelsize=leaf_font_size)


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
    Read a whitespace-separated rectangular grid (NxM).
    Returns: GRID, WORD_TO_POS, WORD_TO_TID, ID_TO_WORD, ROW_LABELS
    """
    rows = []
    with path.open("r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split())

    if not rows:
        raise ValueError("Grid file is empty")

    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(
            "Expected rectangular grid, got row lengths "
            f"{[len(r) for r in rows]}"
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
    ROW_LABELS = [f"Row {r}: " + " ".join(GRID[r]) for r in range(len(GRID))]

    return GRID, WORD_TO_POS, WORD_TO_TID, ID_TO_WORD, ROW_LABELS


def load_embeddings_pt(pt_path: Path, emb_key="embeddings", ids_key="input_ids"):
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


def load_embeddings_batch(
    emb_dir: Path,
    ctx: int,
    NW: int = 50,
    emb_key: str = "embeddings_last",
    ids_key: str = "input_ids_last",
    layer: int = 26,
):
    """
    Load **all** batch files for a given context length from *emb_dir*.

    Auto-discovers files matching ``*_{ctx}_line*_layer{layer}.pt``, loads
    each one, takes the last *NW* tokens, and concatenates across batch
    elements.

    Returns (ids_np, emb_np) with shapes ``(B*W,)`` and ``(B*W, d)``,
    where ``W = min(NW, L_b)`` for each batch element.
    """
    pattern = f"*_{ctx}_line*_layer{layer}.pt"
    files = sorted(emb_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern} in {emb_dir}"
        )

    all_ids = []
    all_emb = []
    for fpath in files:
        obj = torch.load(fpath, map_location="cpu", weights_only=False)
        ids = obj[ids_key]
        emb = obj[emb_key]

        if ids.ndim == 2 and ids.shape[0] == 1:
            ids = ids[0]
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]

        L_b = ids.shape[0]
        W_b = min(NW, L_b)
        ids = ids[-W_b:]
        emb = emb[-W_b:]

        all_ids.append(ids)
        all_emb.append(emb.float())

    all_ids = torch.cat(all_ids, dim=0)
    all_emb = torch.cat(all_emb, dim=0)

    B = len(files)
    print(
        f"Loaded batch: B={B}, NW={NW}, "
        f"total pooled={all_ids.shape[0]}, d={all_emb.shape[1]}"
    )
    return all_ids.numpy(), all_emb.numpy()

# ── Circle structure parsers ────────────────────────────────────────

def load_circle_structure(path: Path, token_to_id: dict, encoding="utf-8"):
    """
    Parse a flat circle structure like:

    1 2 3 4 5 ... N

    Returns:
        tokens,
        token_to_group (unique index per token),
        edges (cyclic),
        WORD_TO_TID,
        ID_TO_WORD,
        group_labels
    """
    lines = path.read_text(encoding=encoding).strip().splitlines()

    tokens = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens.extend(line.split())

    if not tokens:
        raise ValueError("Circle structure file is empty")

    missing = [t for t in tokens if t not in token_to_id]
    if missing:
        raise KeyError(f"Missing tokens: {missing[:8]}")

    WORD_TO_TID = {t: token_to_id[t] for t in tokens}
    ID_TO_WORD = {tid: t for t, tid in WORD_TO_TID.items()}

    # --- cyclic edges ---
    edges = []
    n = len(tokens)
    for i in range(n):
        t1 = WORD_TO_TID[tokens[i]]
        t2 = WORD_TO_TID[tokens[(i + 1) % n]]  # wrap around
        edges.append((min(t1, t2), max(t1, t2)))

    # --- each token is its own group ---
    token_to_group = {t: i for i, t in enumerate(tokens)}
    group_labels = {i: f"{tokens[i]}" for i in range(n)}

    return tokens, token_to_group, edges, WORD_TO_TID, ID_TO_WORD, group_labels

# ── Tree structure parsers ────────────────────────────────────────────


def load_tree_structure(path: Path, token_to_id: dict, encoding="utf-8"):
    """
    Parse a tree structure.
    Only keep nodes present in token_to_id (e.g. leaves).
    Internal structural nodes are ignored in the final graph.

    Returns:
        tokens (only valid tokens),
        token_to_depth,
        edges,
        WORD_TO_TID,
        ID_TO_WORD,
        depth_labels
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

        parent_at_depth[depth] = token

        # Only keep tokens that exist in vocab
        if token in token_to_id:
            tokens.append(token)
            token_to_depth[token] = depth

            # Check if parent exists AND is a valid token
            if depth > 0:
                parent = parent_at_depth.get(depth - 1)
                if parent in token_to_id:
                    edges_words.append((parent, token))

    if not tokens:
        raise ValueError("No valid tokens found in tree file.")

    WORD_TO_TID = {t: token_to_id[t] for t in tokens}
    ID_TO_WORD = {tid: t for t, tid in WORD_TO_TID.items()}

    edges = [
        (min(WORD_TO_TID[p], WORD_TO_TID[c]), max(WORD_TO_TID[p], WORD_TO_TID[c]))
        for p, c in edges_words
        if p in WORD_TO_TID and c in WORD_TO_TID
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

# def _window_and_filter(all_ids_np, all_emb_np, ID_TO_WORD, NW=np.inf, k=None, title=""):
#     """
#     Take a window of tokens and filter unknown IDs.

#     Modes:
#     1. Window mode (default): take the last NW tokens.
#     2. Per-token mode: if k is given, take the last k occurrences of each unique token.

#     Args:
#         all_ids_np (np.ndarray): Array of token IDs.
#         all_emb_np (np.ndarray): Array of embeddings corresponding to token IDs.
#         ID_TO_WORD (dict): Mapping from token IDs to words.
#         NW (int, optional): Window size for the last NW tokens. Default is np.inf.
#         k (int, optional): If provided, take last k occurrences per token instead of a fixed window.
#         title (str, optional): Title for warning messages.

#     Returns:
#         filtered_ids, filtered_emb: Arrays of IDs and embeddings after filtering.
#     """
#     known_tids = set(ID_TO_WORD.keys())

#     if k is not None:
#         # Per-token mode: take last k occurrences of each token
#         filtered_ids_list = []
#         filtered_emb_list = []

#         for tid in known_tids:
#             # Find indices of this token
#             idxs = np.where(all_ids_np == tid)[0]
#             if len(idxs) == 0:
#                 continue
#             if len(idxs) < k:
#                 print(f"  [{title}] Warning: token '{ID_TO_WORD[tid]}' has only {len(idxs)} occurrences (requested {k})")
#             selected_idxs = idxs[-k:]  # last k occurrences (or fewer)
#             filtered_ids_list.append(all_ids_np[selected_idxs])
#             filtered_emb_list.append(all_emb_np[selected_idxs])

#         if filtered_ids_list:
#             filtered_ids = np.concatenate(filtered_ids_list)
#             filtered_emb = np.concatenate(filtered_emb_list)
#         else:
#             filtered_ids = np.array([], dtype=all_ids_np.dtype)
#             filtered_emb = np.array([], dtype=all_emb_np.dtype)
#     else:
#         # Window mode: take last NW tokens
#         L = len(all_ids_np)
#         W = min(NW, L)
#         win_ids = all_ids_np[-W:]
#         win_emb = all_emb_np[-W:]

#         keep_mask = np.isin(win_ids, list(known_tids))
#         n_discard = int((~keep_mask).sum())
#         if n_discard > 0:
#             print(f"  [{title}] Discarded {n_discard}/{len(win_ids)} embeddings with unknown token IDs")

#         filtered_ids = win_ids[keep_mask]
#         filtered_emb = win_emb[keep_mask]

#     return filtered_ids, filtered_emb

def _window_and_filter(
    all_ids_np,
    all_emb_np,
    ID_TO_WORD,
    NW=np.inf,
    k=None,
    title="",
    t_start=None,
    t_end=None,
):
    """
    Take a domain slice of tokens and filter unknown IDs.

    Pipeline:
    1) Restrict to domain [t_start:t_end] (Python slicing semantics: start inclusive, end exclusive)
    2) Apply one of:
       - Window mode (default): take the last NW tokens within the domain
       - Per-token mode (if k is given): take the last k occurrences of each unique token within the domain
    3) Filter unknown token IDs

    Args:
        all_ids_np (np.ndarray): Array of token IDs, shape (T,)
        all_emb_np (np.ndarray): Array of embeddings, shape (T, d)
        ID_TO_WORD (dict): Mapping from token IDs to words.
        NW (int or np.inf, optional): Window size for the last NW tokens (within the domain).
        k (int, optional): If provided, take last k occurrences per token (within the domain).
        title (str, optional): Title for warning messages.
        t_start (int or None, optional): Start index of the domain (inclusive). Default None -> 0.
        t_end (int or None, optional): End index of the domain (exclusive). Default None -> len(all_ids_np).

    Returns:
        filtered_ids, filtered_emb: Arrays of IDs and embeddings after filtering.
    """
    # --- basic checks ---
    if len(all_ids_np) != len(all_emb_np):
        raise ValueError(
            f"Length mismatch: len(all_ids_np)={len(all_ids_np)} vs len(all_emb_np)={len(all_emb_np)}"
        )

    T = len(all_ids_np)

    # --- normalize slice bounds (Python-like semantics) ---
    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = T

    # support negative indices like Python slicing
    if t_start < 0:
        t_start = T + t_start
    if t_end < 0:
        t_end = T + t_end

    # clamp to valid range
    t_start = max(0, min(t_start, T))
    t_end = max(0, min(t_end, T))

    if t_end < t_start:
        raise ValueError(f"Invalid slice: t_end ({t_end}) < t_start ({t_start})")

    # --- restrict domain first ---
    dom_ids = all_ids_np[t_start:t_end]
    dom_emb = all_emb_np[t_start:t_end]

    known_tids = set(ID_TO_WORD.keys())

    if k is not None:
        # Per-token mode: take last k occurrences of each token within the domain
        filtered_ids_list = []
        filtered_emb_list = []

        for tid in known_tids:
            idxs = np.where(dom_ids == tid)[0]
            if len(idxs) == 0:
                continue
            if len(idxs) < k:
                print(
                    f"  [{title}] Warning: token '{ID_TO_WORD[tid]}' has only {len(idxs)} occurrences "
                    f"in [{t_start}:{t_end}] (requested {k})"
                )
            selected_idxs = idxs[-k:]  # last k occurrences in the restricted domain
            filtered_ids_list.append(dom_ids[selected_idxs])
            filtered_emb_list.append(dom_emb[selected_idxs])

        if filtered_ids_list:
            filtered_ids = np.concatenate(filtered_ids_list)
            filtered_emb = np.concatenate(filtered_emb_list)
        else:
            filtered_ids = np.array([], dtype=all_ids_np.dtype)
            # preserva shape embeddings vuoto
            filtered_emb = np.empty((0, *all_emb_np.shape[1:]), dtype=all_emb_np.dtype)

    else:
        # Window mode: take last NW tokens within the domain
        L = len(dom_ids)
        if np.isinf(NW):
            W = L
        else:
            W = min(int(NW), L)

        win_ids = dom_ids[-W:]
        win_emb = dom_emb[-W:]

        keep_mask = np.isin(win_ids, list(known_tids))
        n_discard = int((~keep_mask).sum())
        if n_discard > 0:
            print(
                f"  [{title}] Discarded {n_discard}/{len(win_ids)} embeddings with unknown token IDs "
                f"in domain [{t_start}:{t_end}]"
            )

        filtered_ids = win_ids[keep_mask]
        filtered_emb = win_emb[keep_mask]

    return filtered_ids, filtered_emb


def centroids_by_token(ids, emb):
    id_to_embs = {}
    for id, emb in zip(ids, emb):
        if id not in id_to_embs:
            id_to_embs[id] = []
        id_to_embs[id].append(emb)

    #compute centroids
    centroids = []
    filtered_ids_centroids = []
    for id, embs in id_to_embs.items():
        centroid = np.mean(embs, axis=0)
        centroids.append(centroid)
        filtered_ids_centroids.append(id)
    return np.array(filtered_ids_centroids), np.array(centroids)

