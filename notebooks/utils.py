import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import re
import pandas as pd
import seaborn as sns
from collections import defaultdict



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

def load_graph_from_txt(path, kind="auto", diagonals=False, directed_tree=False):
    text = Path(path).read_text(encoding="utf-8")
    
    if kind == "auto":
        stripped = text.lstrip()
        if stripped.startswith("(griglia)"):
            kind = "grid"
        elif "├──" in text or "└──" in text:
            kind = "tree"
        else:
            raise ValueError("Impossibile inferire il formato. Usa kind='grid' o kind='tree'.")
    
    if kind == "grid":
        rows = parse_grid_text(text)
        return grid_to_graph(rows, diagonals=diagonals)
    elif kind == "tree":
        root, edges = parse_tree_text(text)
        return tree_to_graph(root, edges, directed=directed_tree)
    else:
        raise ValueError(f"Formato non supportato: {kind}")

def dirichlet_energy_laplacian(G, nodes, X, weight="weight"):
    """
    E = Tr(X^T L X)
    """
    L = nx.laplacian_matrix(G, nodelist=nodes, weight=weight)  # sparse matrix
    LX = L @ X
    E = float(np.trace(X.T @ LX))
    return E/np.linalg.norm(X, 'fro')**2  # normalizzazione per rendere l'energia indipendente dalla scala di X

# def dirichlet_energy_edgewise(G, nodes, X, weight_attr="weight"):
#     """
#     E = 1/2 sum_{(i,j)} w_ij ||x_i - x_j||^2
#     Per grafi non orientati in networkx ogni arco appare una sola volta,
#     quindi puoi anche omettere il 1/2 a seconda della convenzione.
#     Qui uso la convenzione con somma su archi una sola volta -> NO 1/2.
#     """
#     idx = {node: k for k, node in enumerate(nodes)}
#     E = 0.0
#     for u, v, data in G.edges(data=True):
#         w = float(data.get(weight_attr, 1.0))
#         diff = X[idx[u]] - X[idx[v]]
#         E += w * float(np.dot(diff, diff))
#     return E

def normalize_token_for_graph(s: str, lowercase: bool = True) -> str:
    """
    Normalizza una stringa token del modello per confrontarla con i nodi del grafo.
    Gestisce:
    - spazi iniziali/finali
    - marker SentencePiece (▁)
    - marker GPT2-BPE (Ġ)
    - marker newline speciali (Ċ) -> \n (poi strip)
    """
    if s is None:
        return ""

    s = str(s)

    # Marker comuni dei tokenizer
    s = s.replace("▁", " ")   # sentencepiece word boundary
    s = s.replace("Ġ", " ")   # GPT2 BPE word boundary
    s = s.replace("Ċ", "\n")  # GPT2 newline marker (se presente)

    # strip spazi/newline attorno
    s = s.strip()

    # opzionale: lowercase per matching robusto
    if lowercase:
        s = s.lower()

    return s


def build_graph_node_lookup(G, lowercase: bool = True):
    """
    Costruisce una mappa normalized_label -> original_node_label.
    Se hai nodi duplicati dopo normalizzazione, segnala warning.
    """
    lookup = {}
    collisions = {}

    for n in G.nodes():
        key = normalize_token_for_graph(n, lowercase=lowercase)
        if key in lookup and lookup[key] != n:
            collisions.setdefault(key, set()).update([lookup[key], n])
        else:
            lookup[key] = n

    if collisions:
        print("[WARN] Collisioni dopo normalizzazione (stesso key per nodi diversi):")
        for k, vals in list(collisions.items())[:10]:
            print(f"  {k!r} -> {sorted(vals)}")
        if len(collisions) > 10:
            print(f"  ... ({len(collisions)-10} altre)")

    return lookup


def get_record_observed_token(rec):
    """
    Compatibilità formato nuovo/vecchio.
    """
    return rec.get("input_token_str", rec.get("last_input_token_str", ""))


def get_record_position(rec):
    pos1 = rec.get("token_position_1based")
    if pos1 is None and rec.get("last_token_pos") is not None:
        pos1 = rec["last_token_pos"] + 1
    return pos1


def compute_rule_following_mass_for_record(
    rec,
    G,
    graph_lookup=None,
    topk_field="top_decoded",
    probs_field="top_probs",
    lowercase=True,
):
    """
    Restituisce un dict con:
    - observed token normalizzato
    - vicini validi (normalizzati)
    - rule_following_mass@k = somma probs delle predizioni che sono vicini nel grafo
    - matched predictions
    """
    if graph_lookup is None:
        graph_lookup = build_graph_node_lookup(G, lowercase=lowercase)

    observed_raw = get_record_observed_token(rec)
    observed_norm = normalize_token_for_graph(observed_raw, lowercase=lowercase)

    # Se il token osservato non è un nodo del grafo (dopo normalizzazione), non possiamo valutare
    if observed_norm not in graph_lookup:
        return {
            "ok": False,
            "reason": "observed_token_not_in_graph",
            "observed_raw": observed_raw,
            "observed_norm": observed_norm,
            "rule_following_mass": np.nan,
            "matched": [],
            "neighbors_norm": [],
        }

    observed_node = graph_lookup[observed_norm]

    # vicini nel grafo (etichette originali) -> normalizzati
    neighbors = list(G.neighbors(observed_node))
    neighbors_norm = {
        normalize_token_for_graph(n, lowercase=lowercase)
        for n in neighbors
    }
    neighbors_norm.discard("")  # pulizia eventuale

    preds = rec.get(topk_field) or rec.get("top_tokens", [])
    probs = rec.get(probs_field, [])

    matched = []
    mass = 0.0

    for pred_raw, p in zip(preds, probs):
        pred_norm = normalize_token_for_graph(pred_raw, lowercase=lowercase)
        if pred_norm in neighbors_norm:
            mass += float(p)
            matched.append({
                "pred_raw": pred_raw,
                "pred_norm": pred_norm,
                "prob": float(p),
            })

    return {
        "ok": True,
        "reason": None,
        "observed_raw": observed_raw,
        "observed_norm": observed_norm,
        "observed_node": observed_node,
        "neighbors_norm": sorted(neighbors_norm),
        "rule_following_mass": float(mass),
        "matched": matched,
    }


def compute_rule_following_accuracy_over_records(records, G, lowercase=True):
    """
    Calcola la metrica su tutti i record e restituisce:
    - summary
    - dataframe con un record per checkpoint
    """
    graph_lookup = build_graph_node_lookup(G, lowercase=lowercase)

    rows = []
    valid_masses = []

    for rec in records:
        pos1 = get_record_position(rec)
        result = compute_rule_following_mass_for_record(
            rec, G, graph_lookup=graph_lookup, lowercase=lowercase
        )

        row = {
            "token_position_1based": pos1,
            "line_index": rec.get("line_index"),
            "observed_raw": result["observed_raw"],
            "observed_norm": result["observed_norm"],
            "ok": result["ok"],
            "reason": result["reason"],
            "rule_following_mass_topk": result["rule_following_mass"],
            "num_matched_preds": len(result["matched"]),
            "matched_preds": [m["pred_norm"] for m in result["matched"]],
            "matched_probs": [m["prob"] for m in result["matched"]],
            "neighbors_norm": result["neighbors_norm"] if result["ok"] else [],
        }
        rows.append(row)

        if result["ok"] and not math.isnan(result["rule_following_mass"]):
            valid_masses.append(result["rule_following_mass"])

    df = pd.DataFrame(rows).sort_values(
        by=["token_position_1based"], na_position="last"
    ).reset_index(drop=True)

    summary = {
        "num_records_total": len(records),
        "num_records_evaluable": int(df["ok"].sum()) if len(df) else 0,
        "num_records_skipped": int((~df["ok"]).sum()) if len(df) else 0,
        "mean_rule_following_mass_topk": float(np.mean(valid_masses)) if valid_masses else np.nan,
        "median_rule_following_mass_topk": float(np.median(valid_masses)) if valid_masses else np.nan,
    }

    return summary, df

def build_word_centroid_map(centroid_ids, centroids, id_to_word):
    """
    Restituisce dict: parola -> vettore centroid
    """
    word_to_centroid = {}
    for tid, vec in zip(centroid_ids, centroids):
        w = id_to_word.get(int(tid), str(int(tid)))
        word_to_centroid[w] = np.asarray(vec, dtype=np.float64)
    return word_to_centroid

def aggregate_cluster_centroids(G_cluster, cluster_to_words, word_to_centroid, agg="mean"):
    """
    Costruisce X (n_cluster, d) allineato ai nodi di G_cluster.
    Restituisce (cluster_labels, X)
    """
    cluster_labels = list(G_cluster.nodes())
    X = []

    missing = {}

    for c in cluster_labels:
        if c not in cluster_to_words:
            raise KeyError(f"Cluster {c!r} non presente in cluster_to_words")

        words = cluster_to_words[c]
        vecs = []
        miss = []

        for w in words:
            if w in word_to_centroid:
                vecs.append(word_to_centroid[w])
            else:
                miss.append(w)

        if miss:
            missing[c] = miss

        if len(vecs) == 0:
            raise ValueError(f"Nessun embedding disponibile per cluster {c!r} (words={words})")

        M = np.stack(vecs, axis=0)
        if agg == "mean":
            xc = M.mean(axis=0)
        elif agg == "sum":
            xc = M.sum(axis=0)
        else:
            raise ValueError("agg deve essere 'mean' o 'sum'")

        X.append(xc)

    if missing:
        print("Attenzione: parole mancanti in alcuni cluster:")
        for c, miss in missing.items():
            print(f"  {c}: {miss}")

    X = np.stack(X, axis=0).astype(np.float64)
    return cluster_labels, X


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