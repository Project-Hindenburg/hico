import random
from typing import List, Tuple, Dict, Optional

class WordGrid:
    # Words are arranged in a 2D grid, and transitions can only happen to adjacent cells (up/down/left/right). Optionally, the grid can wrap around like a torus.
    def __init__(
        self,
        grid: List[List[str]],
        torus: bool = False,
        transition_probs: Optional[
            Dict[str, Dict[str, float]]
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
        # Construct adjacency list for each cell coordinates based on grid structure and torus setting
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
                print(f"Warning: source word '{src_word}' not found in grid, skipping this probability.")
                continue

            src = self.word_to_pos[src_word]
            neighbors = set(self.adjacency[src])

            valid = {}
            for tgt_word, weight in targets.items():
                if tgt_word not in self.word_to_pos:
                    print(f"Warning: target word '{tgt_word}' not found in grid, skipping this probability.")
                    continue

                tgt = self.word_to_pos[tgt_word]

                if tgt not in neighbors:
                    print(f"ERROR: Invalid transition '{src_word}' -> '{tgt_word}' (not adjacent).")
                    continue

                if weight <= 0:
                    print(f"ERROR: Non-positive weight for '{src_word}' -> '{tgt_word}'.")
                    continue

                valid[tgt] = weight


            if valid:
                cleaned[src] = valid

        self.transition_probs = cleaned


    def _choose_next(self, current, rng):
        neighbors = self.adjacency[current]

        probs = self.transition_probs.get(current)

        if probs is None:
            return rng.choice(neighbors)

        weighted = [(n, probs.get(n, 0.0)) for n in neighbors]

        total = sum(w for _, w in weighted)
        if total == 0:
            return rng.choice(neighbors)

        r = rng.random() * total
        acc = 0.0
        for n, w in weighted:
            acc += w
            if r <= acc:
                return n

        return neighbors[-1]


    def generate_sequence(self, length, start=None, rng=None):
        if self.rows == 0 or self.cols == 0:
            return []

        if rng is None:
            rng = random.Random(0)

        if start is None:
            start = (
                rng.randrange(self.rows),
                rng.randrange(self.cols)
            )
        elif isinstance(start, int):
            start = (start // self.cols, start % self.cols)

        current = start
        sequence = [self.grid[current[0]][current[1]]]

        for _ in range(length - 1):
            current = self._choose_next(current, rng)
            r, c = current
            sequence.append(self.grid[r][c])

        return sequence


    def print_grid(self):
        print("Graph structure:")
        for row in self.grid:
            print("  ".join(f"{word:<{self.longest_word_length}}" for word in row))
        print()

    def save_grid(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for row in self.grid:
                f.write("  ".join(f"{word:<{self.longest_word_length}}" for word in row) + "\n")


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
                print(f"Warning: source word '{src_word}' not found in tree, skipping this probability.")
                continue

            src = self.word_to_index[src_word]
            neighbors = set(self.edges[src])

            valid = {}
            for tgt_word, weight in targets.items():
                if tgt_word not in self.word_to_index:
                    print(f"Warning: target word '{tgt_word}' not found in tree, skipping this probability.")
                    continue

                tgt = self.word_to_index[tgt_word]
                if tgt not in neighbors:
                    print(f"ERROR: Invalid transition '{src_word}' -> '{tgt_word}' (not parent/child).")
                    continue

                if weight <= 0:
                    print(f"ERROR: Non-positive weight for '{src_word}' -> '{tgt_word}'.")
                    continue

                valid[tgt] = weight

            if valid:
                cleaned[src] = valid

        self.transition_probs = cleaned


    # -------------------------
    # Sampling logic
    # -------------------------

    def _choose_next(self, current: int, rng: random.Random) -> int:
        neighbors = self.edges[current]
        probs = self.transition_probs.get(current)

        if probs is None:
            return rng.choice(neighbors)

        weighted = [(n, probs.get(n, 0.0)) for n in neighbors]
        total = sum(w for _, w in weighted)

        if total == 0:
            return rng.choice(neighbors)

        r = rng.random() * total
        acc = 0.0
        for n, w in weighted:
            acc += w
            if r <= acc:
                return n

        return neighbors[-1]

    def generate_sequence(
        self,
        length: int,
        start: Optional[int] = None,
        rng: Optional[random.Random] = None
    ) -> List[str]:
        if not self.words:
            return []

        if rng is None:
            rng = random.Random(0)

        if start is None:
            start = rng.randrange(len(self.words))

        current = start
        seq = [self.words[current]]

        for _ in range(length - 1):
            current = self._choose_next(current, rng)
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
    
    def save_tree(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            root = 0

            def dfs(node: int, parent: Optional[int], prefix: str, is_last: bool):
                connector = "└── " if is_last else "├── "
                f.write(prefix + connector + self.words[node] + "\n")

                children = [n for n in self.edges[node] if n != parent]
                for i, child in enumerate(children):
                    last = (i == len(children) - 1)
                    extension = "    " if is_last else "│   "
                    dfs(child, node, prefix + extension, last)

            f.write(self.words[root] + "\n")
            children = self.edges[root]
            for i, child in enumerate(children):
                dfs(child, root, "", i == len(children) - 1)


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
                if tgt not in neighbors:
                    print(f"ERROR: Invalid cluster transition {src_word} -> {tgt_word} (not parent/child).")
                    continue

                if weight <= 0:
                    print(f"ERROR: Non-positive weight for {src_word} -> {tgt_word}.")
                    continue

                valid[tgt] = weight


            if valid:
                cleaned[src] = valid

        self.transition_probs = cleaned


    # -------------------------
    # Sampling logic
    # -------------------------

    def _choose_next(self, current: int, rng: random.Random) -> int:
        neighbors = self.edges[current]
        probs = self.transition_probs.get(current)

        if probs is None:
            return rng.choice(neighbors)

        weighted = [(n, probs.get(n, 0.0)) for n in neighbors]
        total = sum(w for _, w in weighted)

        if total == 0:
            return rng.choice(neighbors)

        r = rng.random() * total
        acc = 0.0
        for n, w in weighted:
            acc += w
            if r <= acc:
                return n

        return neighbors[-1]

    def generate_sequence(
        self,
        length: int,
        start: Optional[int] = None,
        rng = None
    ) -> List[str]:
        if not self.clusters:
            return []

        if rng is None:
            rng = random

        if start is None:
            start = rng.randrange(len(self.clusters))
        current = start
        seq = [rng.choice(self.clusters[current])]

        for _ in range(length - 1):
            current = self._choose_next(current, rng)
            seq.append(rng.choice(self.clusters[current]))

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

    def save_tree(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            root = 0

            def dfs(node: int, parent: Optional[int], prefix: str, is_last: bool):
                connector = "└── " if is_last else "├── "
                f.write(prefix + connector + "(" + ", ".join(self.clusters[node]) + ")\n")

                children = [n for n in self.edges[node] if n != parent]
                for i, child in enumerate(children):
                    last = (i == len(children) - 1)
                    extension = "    " if is_last else "│   "
                    dfs(child, node, prefix + extension, last)

            f.write("(" + ", ".join(self.clusters[root]) + ")\n")
            children = self.edges[root]
            for i, child in enumerate(children):
                dfs(child, root, "", i == len(children) - 1)

