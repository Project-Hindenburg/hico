import random
from typing import List, Tuple, Dict, Optional

class WordGrid:
    # Words are arranged in a 2D grid, and transitions can only happen to adjacent cells (up/down/left/right). Optionally, the grid can wrap around like a torus.
    def __init__(
        self,
        grid: List[List[str]],
        torus: bool = False,
        transition_probs: Optional[
            Dict[Tuple[int, int], Dict[Tuple[int, int], float]]
        ] = None
    ):
        """
        transition_probs:
            Optional dict mapping:
            (r, c) -> {(nr, nc): probability, ...}

            Probabilities do NOT need to be normalized.
            Missing entries default to uniform distribution.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.torus = torus
        self.transition_probs = transition_probs or {}

        # Build adjacency first
        self.adjacency = self._build_adjacency()

        # Build word → position mapping
        self.word_to_pos = {}
        for r in range(self.rows):
            for c in range(self.cols):
                self.word_to_pos[self.grid[r][c]] = (r, c)

        # Now it's safe to sanitize probabilities
        self._sanitize_transition_probs()

        self.longest_word_length = max(
            len(word) for row in grid for word in row
        )


    def _build_adjacency(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        adj = {}

        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = []

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc

                    if self.torus:
                        nr %= self.rows
                        nc %= self.cols
                        neighbors.append((nr, nc))
                    else:
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            neighbors.append((nr, nc))

                adj[(r, c)] = neighbors

        return adj
    
    def _sanitize_transition_probs(self):
        cleaned = {}

        for src_word, targets in self.transition_probs.items():
            if src_word not in self.word_to_pos:
                continue

            src = self.word_to_pos[src_word]
            neighbors = set(self.adjacency[src])

            valid = {}
            for tgt_word, weight in targets.items():
                if tgt_word not in self.word_to_pos:
                    continue

                tgt = self.word_to_pos[tgt_word]
                if tgt in neighbors and weight > 0:
                    valid[tgt] = weight

            if valid:
                cleaned[src] = valid

        self.transition_probs = cleaned


    def _choose_next(self, current: Tuple[int, int]) -> Tuple[int, int]:
        neighbors = self.adjacency[current]

        probs = self.transition_probs.get(current)

        # Default: uniform distribution
        if probs is None:
            return random.choice(neighbors)

        # Filter only valid neighbors
        weighted = [(n, probs.get(n, 0.0)) for n in neighbors]

        total = sum(w for _, w in weighted)
        if total == 0:
            # fallback safety
            return random.choice(neighbors)

        r = random.random() * total
        acc = 0.0
        for n, w in weighted:
            acc += w
            if r <= acc:
                return n

        return neighbors[-1]  # numerical safety

    def generate_sequence(
        self,
        length: int,
        start: Tuple[int, int] = None
    ) -> List[str]:
        if self.rows == 0 or self.cols == 0:
            return []

        if start is None:
            start = (
                random.randrange(self.rows),
                random.randrange(self.cols)
            )

        current = start
        sequence = [self.grid[current[0]][current[1]]]

        for _ in range(length - 1):
            current = self._choose_next(current)
            r, c = current
            sequence.append(self.grid[r][c])

        return sequence

    def print_grid(self):
        print("Graph structure:")
        for row in self.grid:
            print("  ".join(f"{word:<{self.longest_word_length}}" for word in row))
        print()


class WordTree:
    # Words are arranged in a tree structure, where each level represents a different "layer" of the tree. Transitions can only happen between parent and child nodes. The tree can be binary, ternary, etc., depending on the max_children parameter.
    def __init__(
        self,
        levels: List[List[str]],
        max_children: int = 2,
        transition_probs: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        levels:
            List of tree levels, e.g.
            [
              ["root"],
              ["c1", "c2"],
              ["c3", "c4", "c5", "c6"]
            ]

        max_children:
            Maximum number of children per node

        transition_probs:
            Optional dict:
            node_index -> {neighbor_index: weight}
        """
        self.levels = levels
        self.max_children = max_children
        self.transition_probs = transition_probs or {}

        self.words: List[str] = []
        self.edges: Dict[int, List[int]] = {}

        self._build_tree()

    # -------------------------
    # Tree construction
    # -------------------------

    def _build_tree(self):
        # Flatten words and keep index mapping
        index = 0
        level_indices: List[List[int]] = []

        for level in self.levels:
            idxs = []
            for word in level:
                self.words.append(word)
                self.edges[index] = []
                idxs.append(index)
                index += 1
            level_indices.append(idxs)

        self.word_to_index = {
            word: i for i, word in enumerate(self.words)
        }

        # Connect levels
        for lvl in range(len(level_indices) - 1):
            parents = level_indices[lvl]
            children = level_indices[lvl + 1]

            child_ptr = 0
            for p in parents:
                for _ in range(self.max_children):
                    if child_ptr >= len(children):
                        break
                    c = children[child_ptr]
                    self.edges[p].append(c)
                    self.edges[c].append(p)
                    child_ptr += 1
        self._sanitize_transition_probs()

    
    def _sanitize_transition_probs(self):
        """
        Convert word-based transition probabilities into
        index-based ones, keeping only valid tree edges.
        """
        cleaned = {}

        if not self.transition_probs:
            self.transition_probs = {}
            return

        for src_word, targets in self.transition_probs.items():
            if src_word not in self.word_to_index:
                continue

            src = self.word_to_index[src_word]
            neighbors = set(self.edges[src])

            valid = {}
            for tgt_word, weight in targets.items():
                if tgt_word not in self.word_to_index:
                    continue

                tgt = self.word_to_index[tgt_word]
                if tgt in neighbors and weight > 0:
                    valid[tgt] = weight

            if valid:
                cleaned[src] = valid

        self.transition_probs = cleaned


    # -------------------------
    # Sampling logic
    # -------------------------

    def _choose_next(self, current: int) -> int:
        neighbors = self.edges[current]
        probs = self.transition_probs.get(current)

        if probs is None:
            return random.choice(neighbors)

        weighted = [(n, probs.get(n, 0.0)) for n in neighbors]
        total = sum(w for _, w in weighted)

        if total == 0:
            return random.choice(neighbors)

        r = random.random() * total
        acc = 0.0
        for n, w in weighted:
            acc += w
            if r <= acc:
                return n

        return neighbors[-1]

    def generate_sequence(
        self,
        length: int,
        start: Optional[int] = None
    ) -> List[str]:
        if not self.words:
            return []

        if start is None:
            start = random.randrange(len(self.words))

        current = start
        seq = [self.words[current]]

        for _ in range(length - 1):
            current = self._choose_next(current)
            seq.append(self.words[current])

        return seq

    # -------------------------
    # Debug / visualization
    # -------------------------

    def print_tree(self):
        root = 0
        print("Graph structure:")

        def dfs(node: int, parent: Optional[int], prefix: str, is_last: bool):
            connector = "└── " if is_last else "├── "
            print(prefix + connector + self.words[node])

            children = [n for n in self.edges[node] if n != parent]
            for i, child in enumerate(children):
                last = (i == len(children) - 1)
                extension = "    " if is_last else "│   "
                dfs(child, node, prefix + extension, last)

        print(self.words[root])
        children = self.edges[root]
        for i, child in enumerate(children):
            dfs(child, root, "", i == len(children) - 1)
        print()


class WordTreeCluster:
    # Variation of WordTree where each node is a cluster of words, and edges represent allowed transitions between clusters. This allows for more flexible structures that aren't strictly binary/ternary trees, but still maintain a hierarchical organization.
    def __init__(
        self,
        levels: List[List[Tuple[str, ...]]],
        max_children: int = 2,
        transition_probs: Optional[Dict[Tuple[str, ...], Dict[Tuple[str, ...], float]]] = None
    ):
        """
        levels:
            List of tree levels, e.g.
            [
              [("root",)],
              [("c1",), ("c2",)],
              [("c3",), ("c4",), ("c5",), ("c6",)]
            ]

        max_children:
            Maximum number of children per node

        transition_probs:
            Optional dict:
            node_index -> {neighbor_index: weight}
        """
        self.levels = levels
        self.max_children = max_children
        self.transition_probs = transition_probs or {}

        self.clusters: List[Tuple[str, ...]] = []
        self.edges: Dict[int, List[int]] = {}

        self._build_tree()

    # -------------------------
    # Tree construction
    # -------------------------

    def _build_tree(self):
        # Flatten words and keep index mapping
        index = 0
        level_indices: List[List[int]] = []

        for level in self.levels:
            idxs = []
            for cluster in level:
                self.clusters.append(cluster)
                self.edges[index] = []
                idxs.append(index)
                index += 1
            level_indices.append(idxs)

        self.cluster_to_index = {
            cluster: i for i, cluster in enumerate(self.clusters)
        }

        # Connect levels
        for lvl in range(len(level_indices) - 1):
            parents = level_indices[lvl]
            children = level_indices[lvl + 1]

            child_ptr = 0
            for p in parents:
                for _ in range(self.max_children):
                    if child_ptr >= len(children):
                        break
                    c = children[child_ptr]
                    self.edges[p].append(c)
                    self.edges[c].append(p)
                    child_ptr += 1
        self._sanitize_transition_probs()

    
    def _sanitize_transition_probs(self):
        """
        Convert word-based transition probabilities into
        index-based ones, keeping only valid tree edges.
        """
        cleaned = {}

        if not self.transition_probs:
            self.transition_probs = {}
            return

        for src_word, targets in self.transition_probs.items():
            if src_word not in self.cluster_to_index:
                continue

            src = self.cluster_to_index[src_word]
            neighbors = set(self.edges[src])

            valid = {}
            for tgt_word, weight in targets.items():
                if tgt_word not in self.cluster_to_index:
                    continue

                tgt = self.cluster_to_index[tgt_word]
                if tgt in neighbors and weight > 0:
                    valid[tgt] = weight

            if valid:
                cleaned[src] = valid

        self.transition_probs = cleaned


    # -------------------------
    # Sampling logic
    # -------------------------

    def _choose_next(self, current: int) -> int:
        neighbors = self.edges[current]
        probs = self.transition_probs.get(current)

        if probs is None:
            return random.choice(neighbors)

        weighted = [(n, probs.get(n, 0.0)) for n in neighbors]
        total = sum(w for _, w in weighted)

        if total == 0:
            return random.choice(neighbors)

        r = random.random() * total
        acc = 0.0
        for n, w in weighted:
            acc += w
            if r <= acc:
                return n

        return neighbors[-1]

    def generate_sequence(
        self,
        length: int,
        start: Optional[int] = None
    ) -> List[str]:
        if not self.clusters:
            return []

        if start is None:
            start = random.randrange(len(self.clusters))
        current = start
        seq = [random.choice(self.clusters[current])]

        for _ in range(length - 1):
            current = self._choose_next(current)
            seq.append(random.choice(self.clusters[current]))


        return seq

    # -------------------------
    # Debug / visualization
    # -------------------------

    def print_tree(self):
        root = 0
        print("Graph structure:")

        def dfs(node: int, parent: Optional[int], prefix: str, is_last: bool):
            connector = "└── " if is_last else "├── "
            print(prefix + connector + "(" + ", ".join(self.clusters[node]) + ")")

            children = [n for n in self.edges[node] if n != parent]
            for i, child in enumerate(children):
                last = (i == len(children) - 1)
                extension = "    " if is_last else "│   "
                dfs(child, node, prefix + extension, last)

        print("(" + ", ".join(self.clusters[root]) + ")")
        children = self.edges[root]
        for i, child in enumerate(children):
            dfs(child, root, "", i == len(children) - 1)
        print()    

if __name__ == "__main__":
    print("WordGrid Example:")
    grid =[
            ["sand",  "handle",  "math",      "grape" ],
            ["queue", "biscuit", "straw",     "lamp"  ],
            ["birch", "shampoo", "trumpet",   "school"],
            ["quilt", "bishop",  "sprinkler", "bee"   ]
        ]


    print("Normal boundaries:")
    wg1 = WordGrid(grid, torus=False)
    wg1.print_grid()
    print(" -> ".join(wg1.generate_sequence(8)))

    print("\nPac-Man boundaries:")
    wg2 = WordGrid(grid, torus=True)
    wg2.print_grid()
    print(" -> ".join(wg2.generate_sequence(8)))

    transition_probs = {
        # Row 0
        "sand":    {"handle": 0.9, "queue": 0.1},

        # Row 1
        "queue":   {"sand": 1, "biscuit": 0, "birch": 0},
    }

    print("\nWith transition probabilities (from sand prefer handle, from queue always sand):")
    wg = WordGrid(grid, torus=False, transition_probs=transition_probs)
    print(" -> ".join(wg.generate_sequence(8)))

    print("\nWordTree Example:")

    levels = [
        ["grape"],
        ["lamp", "container"],
        ["eye", "bishop", "school", "sprinkler"]
    ]

    tree = WordTree(levels, max_children=2)
    tree.print_tree()
    print(" -> ".join(tree.generate_sequence(8)))

    print("\nTernary Tree:")
    tree3 = WordTree(levels, max_children=3)
    tree3.print_tree()
    print(" -> ".join(tree3.generate_sequence(8)))

    print("\nWordTreeCluster Example:")
    c1 = ("grape", "apple")
    c2 = ("lamp", "lantern")
    c3 = ("container", "box")
    c4 = ("eye", "ear")
    c5 = ("bishop", "knight")
    c6 = ("school", "university")
    c7 = ("sprinkler", "hose")
    cluster_levels = [[c1], [c2, c3], [c4, c5, c6, c7]]
    transition_probs = {
        c1: {c2: 0.8, c3: 0.2},
        c2: {c4: 0.5, c5: 0.5},
        c3: {c6: 1.0}
    }
    tree_cluster = WordTreeCluster(cluster_levels, max_children=2, transition_probs=transition_probs)
    tree_cluster.print_tree()
    print(" -> ".join(tree_cluster.generate_sequence(8)))