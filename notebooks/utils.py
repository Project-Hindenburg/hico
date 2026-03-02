import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import re
import pandas as pd
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import PCA

from vis import (
    Plotter,
    _window_and_filter,
    load_embeddings_pt,
    centroids_by_token,
)

def parse_cycle_text(text: str):
    """
    Estrae i token (nodi) da un file che contiene una sequenza su una o più righe.
    Esempio:
    1  2  3  4  5
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Testo vuoto")

    # Se vuoi accettare anche un'eventuale riga header tipo "(cerchio)"
    if lines and lines[0].lower() in {"(cerchio)", "(cycle)", "(anello)"}:
        lines = lines[1:]
        if not lines:
            raise ValueError("Solo header, nessun nodo trovato")

    # Tokenizza su whitespace
    tokens = []
    for ln in lines:
        tokens.extend(ln.split())

    if len(tokens) < 3:
        raise ValueError("Un ciclo richiede almeno 3 nodi")

    # Nota: i nodi sono stringhe (come già fai in grid/tree). Se vuoi int, converti qui.
    return tokens

def cycle_to_graph(nodes, directed=False):
    """
    Costruisce un grafo (non orientato di default) a cerchio:
    (n0-n1), (n1-n2), ... (n_{k-2}-n_{k-1}), (n_{k-1}-n0)
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # aggiungi nodi (pos opzionale utile per layout circolare)
    for i, n in enumerate(nodes):
        G.add_node(n, idx=i)

    k = len(nodes)
    for i in range(k):
        u = nodes[i]
        v = nodes[(i + 1) % k]
        G.add_edge(u, v, weight=1.0)

    return G





def parse_grid_text(text: str):
    """
    Restituisce una matrice di token (lista di liste), ignorando la riga '(griglia)' se presente.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    
    if lines and lines[0].strip().lower() == "(griglia)":
        lines = lines[1:]
    
    # Split su spazi multipli / whitespace
    rows = [ln.split() for ln in lines]
    
    # Validazione: tutte le righe con stesso numero di colonne
    ncols = len(rows[0])
    for r in rows:
        if len(r) != ncols:
            raise ValueError(f"Riga con numero colonne diverso: atteso {ncols}, trovato {len(r)} -> {r}")
    return rows

def grid_to_graph(rows, diagonals=False):
    """
    Costruisce un grafo non orientato da una griglia di parole.
    Connette vicini ortogonali (e diagonali se diagonals=True).
    """
    G = nx.Graph()
    nrows, ncols = len(rows), len(rows[0])
    
    # aggiungi nodi
    for i in range(nrows):
        for j in range(ncols):
            G.add_node(rows[i][j], pos=(i, j))
    
    # direzioni: destra e sotto bastano per non duplicare archi
    directions = [(0, 1), (1, 0)]
    if diagonals:
        directions += [(1, 1), (1, -1)]
    
    for i in range(nrows):
        for j in range(ncols):
            u = rows[i][j]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < nrows and 0 <= nj < ncols:
                    v = rows[ni][nj]
                    G.add_edge(u, v, weight=1.0)
    
    return G

TREE_BRANCH_RE = re.compile(r"(├──|└──)\s*(.+)$")

def parse_tree_text(text: str):
    """
    Converte un albero testuale Unicode in una lista di archi (parent, child).
    Restituisce (root, edges).
    
    Funziona con righe tipo:
    root
    ├── child1
    │   ├── grandchild
    └── child2
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Testo vuoto")
    
    # prima riga: root (senza prefissi)
    root = lines[0].strip()
    edges = []
    
    # stack: depth -> node_name
    # depth = livello (0=root, 1 figli root, ...)
    stack = {0: root}
    
    for line in lines[1:]:
        # trova la parte con branch + label
        m = TREE_BRANCH_RE.search(line)
        if not m:
            # se il formato non è perfettamente standard, puoi gestire fallback qui
            raise ValueError(f"Riga non riconosciuta come ramo: {line!r}")
        
        label = m.group(2).strip()
        
        # Prefisso prima di ├── / └──
        prefix = line[:m.start()]
        
        # Nei tree unicode standard, ogni livello occupa 4 caratteri:
        # "│   " oppure "    "
        # Quindi la profondità del nodo = (len(prefix) // 4) + 1
        depth = (len(prefix) // 4) + 1
        
        parent_depth = depth - 1
        if parent_depth not in stack:
            raise ValueError(f"Parent non trovato per nodo {label!r} (depth={depth})")
        
        parent = stack[parent_depth]
        edges.append((parent, label))
        
        # aggiorna stack al depth corrente
        stack[depth] = label
        
        # pulizia eventuali depth più profondi non più validi
        to_delete = [d for d in stack if d > depth]
        for d in to_delete:
            del stack[d]
    
    return root, edges

def tree_to_graph(root, edges, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_node(root)
    G.add_edges_from(edges, weight=1.0)
    return G

def load_graph_from_txt(path, kind="auto", diagonals=False, directed_tree=False, directed_cycle=False):
    text = Path(path).read_text(encoding="utf-8")

    if kind == "auto":
        stripped = text.lstrip()
        if stripped.startswith("(griglia)"):
            kind = "grid"
        elif "├──" in text or "└──" in text:
            kind = "tree"
        else:
            # euristica: se sono token su una riga (o più righe) senza struttura tree/grid,
            # interpretalo come ciclo
            kind = "cycle"

    if kind == "grid":
        rows = parse_grid_text(text)
        return grid_to_graph(rows, diagonals=diagonals)

    elif kind == "tree":
        root, edges = parse_tree_text(text)
        return tree_to_graph(root, edges, directed=directed_tree)

    elif kind == "cycle":
        nodes = parse_cycle_text(text)
        return cycle_to_graph(nodes, directed=directed_cycle)

    else:
        raise ValueError(f"Formato non supportato: {kind}")
    
# ----------------------------- methods for loss ----------------------------- #



def normalize_token_for_graph(s: str, lowercase: bool = True) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("▁", " ")
    s = s.replace("Ġ", " ")
    s = s.replace("Ċ", "\n")
    s = s.strip()
    if lowercase:
        s = s.lower()
    return s

def build_graph_node_lookup(G, lowercase: bool = True):
    lookup = {}
    collisions = {}
    for n in G.nodes():
        key = normalize_token_for_graph(n, lowercase=lowercase)
        if key in lookup and lookup[key] != n:
            collisions.setdefault(key, set()).update([lookup[key], n])
        else:
            lookup[key] = n
    if collisions:
        print("[WARN] Collisioni dopo normalizzazione (max 10):")
        for k, vals in list(collisions.items())[:10]:
            print(f"  {k!r} -> {sorted(vals)}")
    return lookup


def compute_neighbor_mass_and_loss_for_record(
    rec,
    G,
    graph_lookup=None,
    topk_field="top_decoded",
    probs_field="top_probs",
    lowercase=True,
):
    if graph_lookup is None:
        graph_lookup = build_graph_node_lookup(G, lowercase=lowercase)

    # NEW: usa anchor_token_str (fallback su vecchi nomi)
    observed_raw = (
        rec.get("anchor_token_str")
        or rec.get("input_token_str")
        or rec.get("last_input_token_str", "")
    )
    observed_norm = normalize_token_for_graph(observed_raw, lowercase=lowercase)

    if observed_norm not in graph_lookup:
        return {
            "ok": False,
            "reason": "observed_token_not_in_graph",
            "observed_raw": observed_raw,
            "observed_norm": observed_norm,
            "mass_topk": np.nan,
            "loss_topk": np.nan,
            "num_neighbors": 0,
            "num_matched": 0,
            "matched": [],
        }

    observed_node = graph_lookup[observed_norm]
    neighbors = list(G.neighbors(observed_node))
    neighbors_norm = {normalize_token_for_graph(n, lowercase=lowercase) for n in neighbors}
    neighbors_norm.discard("")

    preds = rec.get(topk_field) or []
    probs = rec.get(probs_field) or []

    mass = 0.0
    matched = []
    for pred_raw, p in zip(preds, probs):
        pred_norm = normalize_token_for_graph(pred_raw, lowercase=lowercase)
        if pred_norm in neighbors_norm:
            fp = float(p)
            mass += fp
            matched.append({"pred_raw": pred_raw, "pred_norm": pred_norm, "prob": fp})

    loss = float("inf") if mass <= 0.0 else -math.log(mass)

    return {
        "ok": True,
        "reason": None,
        "observed_raw": observed_raw,
        "observed_norm": observed_norm,
        "observed_node": observed_node,
        "neighbors_norm": sorted(neighbors_norm),
        "mass_topk": float(mass),
        "loss_topk": float(loss),
        "num_neighbors": int(len(neighbors_norm)),
        "num_matched": int(len(matched)),
        "matched": matched,
    }


def compute_neighbor_loss_over_records(records, G, topk_field="top_decoded", probs_field="top_probs", lowercase=True):
    graph_lookup = build_graph_node_lookup(G, lowercase=lowercase)
    rows = []
    valid_losses = []

    for rec in records:
        # NEW: usa anchor_pos_1based (fallback su vecchi nomi)
        pos1 = (
            rec.get("anchor_pos_1based")
            or rec.get("token_position_1based")
            or (rec.get("token_position_0based", 0) + 1 if rec.get("token_position_0based") is not None else None)
            or (rec.get("last_token_pos", 0) + 1)
        )

        res = compute_neighbor_mass_and_loss_for_record(
            rec, G,
            graph_lookup=graph_lookup,
            topk_field=topk_field,
            probs_field=probs_field,
            lowercase=lowercase,
        )

        row = {
            "token_position_1based": pos1,
            "line_index": rec.get("line_index"),
            "ok": res["ok"],
            "reason": res["reason"],
            "observed_raw": res["observed_raw"],
            "observed_norm": res["observed_norm"],
            "mass_topk": res["mass_topk"],
            "loss_topk": res["loss_topk"],
            "num_neighbors": res["num_neighbors"],
            "num_matched": res["num_matched"],
            "matched_preds": [m["pred_norm"] for m in res.get("matched", [])],
            "matched_probs": [m["prob"] for m in res.get("matched", [])],
        }
        rows.append(row)

        if res["ok"] and np.isfinite(res["loss_topk"]):
            valid_losses.append(res["loss_topk"])
        else:
            print(f"[WARN] Record non valutabile (pos {pos1}, line {rec.get('line_index')}): {res['reason']} (observed_raw={res['observed_raw']!r})")

    df = pd.DataFrame(rows).sort_values("token_position_1based", na_position="last").reset_index(drop=True)

    summary = {
        "num_records_total": len(records),
        "num_records_evaluable": int(df["ok"].sum()) if len(df) else 0,
        "mean_loss_topk": float(np.mean(valid_losses)) if valid_losses else np.nan,
        "median_loss_topk": float(np.median(valid_losses)) if valid_losses else np.nan,
        "mean_mass_topk": float(np.mean(df.loc[df["ok"], "mass_topk"])) if df["ok"].any() else np.nan,
    }
    return summary, df

def dirichlet_energy_laplacian(G, nodes, X, weight="weight"):
    """
    E = Tr(X^T L X)
    """
    L = nx.laplacian_matrix(G, nodelist=nodes, weight=weight)  # sparse matrix
    LX = L @ X
    E = float(np.trace(X.T @ LX))
    return E/np.linalg.norm(X, 'fro')**2  # normalizzazione per rendere l'energia indipendente dalla scala di X

def process_rw_file(filepath, startpoint=None, endpoint=None, plot=True):
    """
    Reads a random walk txt file.
    Each word is treated as a node.
    Consecutive words form directed edges.

    Returns:
        adj_df         : raw adjacency matrix
        word_counts    : total occurrences per node
        adj_row_prob   : row-normalized transition matrix
    """

    edge_counts = defaultdict(int)
    word_counts = defaultdict(int)
    nodes = set()

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # -------------------------------------------------
            # Case 1: comma-separated format
            # -------------------------------------------------
            if "," in line:
                pairs = [p.strip() for p in line.split(",")]

                if startpoint is not None and len(pairs)*2 > startpoint:
                    startpoint = startpoint // 2
                else:
                    startpoint = 0
                
                if endpoint is not None and len(pairs)*2 > endpoint and endpoint > startpoint:
                    endpoint = endpoint // 2
                else:
                    endpoint = len(pairs)
                
                pairs = pairs[startpoint:endpoint]
                print(f"Processing line with {len(pairs)} pairs, using indices [{startpoint}:{endpoint}]")

                for pair in pairs:
                    words = pair.split()
                    if len(words) != 2:
                        continue

                    src, dst = words
                    edge_counts[(src, dst)] += 1
                    word_counts[src] += 1
                    word_counts[dst] += 1
                    nodes.update([src, dst])

            # -------------------------------------------------
            # Case 2: standard RW format
            # -------------------------------------------------
            else:
                words = line.split()
                if startpoint is not None and len(words) > startpoint:
                    startpoint = startpoint
                else:
                    startpoint = 0
                
                if endpoint is not None and len(words) > endpoint and endpoint > startpoint:
                    endpoint = endpoint
                else:
                    endpoint = len(words)

                words = words[startpoint:endpoint]
                print(f"Processing line with {len(words)} words, using indices [{startpoint}:{endpoint}]")

                for i in range(len(words) - 1):
                    src = words[i]
                    dst = words[i + 1]

                    edge_counts[(src, dst)] += 1
                    word_counts[src] += 1
                    nodes.update([src, dst])

                if words:
                    word_counts[words[-1]] += 1
                    nodes.add(words[-1])

    # -------------------------------------------------
    # Build adjacency matrix
    # -------------------------------------------------
    nodes = sorted(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    adj_matrix = np.zeros((len(nodes), len(nodes)))

    for (src, dst), count in edge_counts.items():
        i = node_index[src]
        j = node_index[dst]
        adj_matrix[i, j] = count

    adj_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)

    # -------------------------------------------------
    # Row-normalized transition matrix
    # -------------------------------------------------
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    adj_row_prob = adj_matrix / row_sums

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            adj_df,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=True
        )
        plt.title("Transition Count Matrix")
        plt.xlabel("To Node")
        plt.ylabel("From Node")
        plt.tight_layout()
        plt.show()

    return adj_df, dict(word_counts), adj_row_prob


def compute_energy(
    VOCAB_PATH: Path,
    GRID_PATH: Path,
    EMB_DIR: Path,
    ctx: int = 2000,
    WIN: int = 10,
    PCA_K: int = 2,
    pc1: int = 0,
    pc2: int = 1,
    K: int = 1,
    MIN_NODES: int | None = None,
    pt_filename: str | None = None,
    graph_kind: str = "tree",
    return_df: bool = False,
    emb_key: str = "embeddings",
    ids_key: str = "input_ids",
):
    """
    Restituisce:
        shuffled_tree_energy = [df_plot["context_len"].values, df_plot["energy"].values]

    Se return_df=True, restituisce anche df_plot.
    """
    if MIN_NODES is None:
        MIN_NODES = PCA_K + 1

    # Setup
    hp_grid = Plotter(VOCAB_PATH, GRID_PATH, EMB_DIR)
    G = load_graph_from_txt(GRID_PATH, kind=graph_kind)
    graph_nodes = list(G.nodes())

    # File embeddings
    if pt_filename is None:
        pt_filename = "reprs_bin_tree_one_rw_2000_line000000_layer32.pt"
    pt_path = EMB_DIR / pt_filename

    ids_np, emb_np = load_embeddings_pt(pt_path, emb_key=emb_key, ids_key=ids_key)

    T_file = len(ids_np)
    t_end_ref = min(ctx, T_file)

    # Finestre crescenti: WIN, 2*WIN, ..., t_end_ref (assicurandolo)
    CONTEXT_LENS = list(range(WIN, t_end_ref + 1, WIN))
    if not CONTEXT_LENS or CONTEXT_LENS[-1] != t_end_ref:
        CONTEXT_LENS.append(t_end_ref)

    all_results = []

    for win_len in CONTEXT_LENS:
        t0 = 0
        t1 = min(win_len, T_file)

        filtered_ids, filtered_emb = _window_and_filter(
            ids_np, emb_np, hp_grid.id_to_word,
            k=K,
            t_start=t0, t_end=t1,
            title=f"ctx={ctx}, win_len={win_len}, local=[{t0}:{t1})"
        )

        if len(filtered_ids) == 0:
            all_results.append({
                "ctx": ctx, "context_len": win_len,
                "t_start_local": t0, "t_end_local": t1,
                "energy": np.nan, "energy_per_edge": np.nan,
                "n_filtered": 0, "n_centroids": 0,
                "n_common": 0, "n_edges_subgraph": 0,
            })
            continue

        centroid_ids, centroids = centroids_by_token(filtered_ids, filtered_emb)

        if len(centroid_ids) == 0:
            all_results.append({
                "ctx": ctx, "context_len": win_len,
                "t_start_local": t0, "t_end_local": t1,
                "energy": np.nan, "energy_per_edge": np.nan,
                "n_filtered": len(filtered_ids), "n_centroids": 0,
                "n_common": 0, "n_edges_subgraph": 0,
            })
            continue

        labels = [hp_grid.id_to_word.get(int(t), str(int(t))) for t in centroid_ids]
        idx = {lab: i for i, lab in enumerate(labels)}
        common = [node for node in graph_nodes if node in idx]

        if len(common) < MIN_NODES:
            E = np.nan
            E_per_edge = np.nan
            m = 0
        else:
            X_common = np.stack([centroids[idx[lab]] for lab in common], axis=0).astype(np.float64)
            X_common = X_common - X_common.mean(axis=0, keepdims=True)

            required_k = max(pc1, pc2) + 1
            max_possible_k = len(common) - 1

            if max_possible_k < required_k:
                E = np.nan
                E_per_edge = np.nan
                m = 0
            else:
                k_pca = min(max(PCA_K, required_k), max_possible_k)

                pca = PCA(n_components=k_pca)
                X_pca_full = pca.fit_transform(X_common)
                X_pca = X_pca_full[:, [pc1, pc2]]

                G_sub = G.subgraph(common).copy()
                m = G_sub.number_of_edges()

                if m == 0:
                    E = np.nan
                    E_per_edge = np.nan
                else:
                    E = dirichlet_energy_laplacian(G_sub, common, X_pca)
                    E_per_edge = E / m

        all_results.append({
            "ctx": ctx, "context_len": win_len,
            "t_start_local": t0, "t_end_local": t1,
            "energy": E, "energy_per_edge": E_per_edge,
            "n_filtered": len(filtered_ids), "n_centroids": len(centroid_ids),
            "n_common": len(common), "n_edges_subgraph": m,
        })

    df = pd.DataFrame(all_results)
    
    print("Total windows:", len(df))
    print("Valid energy windows:", df["energy"].notna().sum())
    print("Dropped windows:", df["energy"].isna().sum())

    df_plot = df.copy().sort_values("context_len")

    shuffled_tree_energy = [df_plot["context_len"].values, df_plot["energy"].values]

    if return_df:
        return shuffled_tree_energy, df_plot
    return shuffled_tree_energy