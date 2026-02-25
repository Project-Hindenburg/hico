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

        # Build adjacency first (for every coordinates, store the list of adjacent coordinates)
        self.adjacency = self._build_adjacency()

        # Build word → position mapping
        # Counting is done over the grid in row-major order, so the first word is at (0, 0), the second at (0, 1), and so on.
        self.word_to_pos = {}
        for r in range(self.rows):
            for c in range(self.cols):
                self.word_to_pos[self.grid[r][c]] = (r, c)

        # Change from word-based to coordinate-based transition probabilities, and validate them against the adjacency list.
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
                
                # For each element of the grid, determine its 4 neighbors (up, down, left, right) and apply torus wrapping if enabled
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

        # Check that all source and target words exist in the grid and that transitions are valid (i.e., between adjacent cells).
        # Convert word-based probabilities to coordinate-based ones.
        for src_word, targets in self.transition_probs.items():
            if src_word not in self.word_to_pos:
                print(f"Warning: source word '{src_word}' not found in grid, skipping this probability.")
                continue
            
            # Take the coordinates of the source word and find its neighbors from the adjacency list
            src = self.word_to_pos[src_word]
            neighbors = set(self.adjacency[src])

            valid = {}
            # Check all target words for the current source word, ensuring they exist in the grid and are valid transitions (i.e., adjacent to the source).
            for tgt_word, weight in targets.items():
                if tgt_word not in self.word_to_pos:
                    print(f"Warning: target word '{tgt_word}' not found in grid, skipping this probability.")
                    continue

                tgt = self.word_to_pos[tgt_word]

                if tgt not in neighbors:
                    print(f"ERROR: Invalid transition '{src_word}' -> '{tgt_word}' (not adjacent).")
                    continue

                if weight < 0:
                    print(f"ERROR: Negative weight for '{src_word}' -> '{tgt_word}'.")
                    continue

                valid[tgt] = weight

            # Only keep valid transitions for this source word src
            if valid:
                cleaned[src] = valid

        self.transition_probs = cleaned


    def _choose_next(self, current, rng):
        # Get the tuple coordinates of the current word and find its neighbors from the adjacency list.
        neighbors = self.adjacency[current]

        # Use the transition probabilities (if any) to choose the next cell among the neighbors.
        probs = self.transition_probs.get(current)

        if probs is None:
            return rng.choice(neighbors)

        weighted = [(n, probs.get(n, 0.0)) for n in neighbors]

        total = sum(w for _, w in weighted)
        if total == 0:
            return rng.choice(neighbors)
        
        # generates a random number between 0 and total
        r = rng.random() * total
        acc = 0.0
        # Iterate through the neighbors and their weights, accumulating the weights until we exceed r, at which point we return the corresponding neighbor.
        # This effectively samples from the weighted distribution defined by the transition probabilities.
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
            # If start is given as a single integer, interpret it as a linear index into the grid
            # (row-major order, ex. 0 is (0, 0), 1 is (0, 1), ..., cols is (1, 0), etc.)
            start = (start // self.cols, start % self.cols)

        current = start
        sequence = [self.grid[current[0]][current[1]]]

        # After first start token, generate the rest of the sequence.
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
        transition_probs: Optional[Dict[str, Dict[str, float]]] = None,
        only_leaves = False
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
        self.only_leaves = only_leaves

        self.words: List[str] = []
        # Edges are stored as an adjacency list: node index -> list of neighbor indices (parents and children)
        self.edges: Dict[int, List[int]] = {}

        self._build_tree()
        self._sanitize_transition_probs()

    # -------------------------
    # Tree construction
    # -------------------------

    def _build_tree(self):
        # Flatten words and keep index mapping
        index = 0
        level_indices: List[List[int]] = []

        # Enumerate nodes through the tree. The root will have idx 0, its children will have indices 1 to max_children, and so on.
        # We also initialize the adjacency list for edges.
        for level in self.levels:
            idxs = []
            for word in level:
                self.words.append(word)
                self.edges[index] = []
                idxs.append(index)
                index += 1
            level_indices.append(idxs)

        # This could probably be moved inside the loop above...
        self.word_to_index = {
            word: i for i, word in enumerate(self.words)
        }

        # Connect levels: we "fill" the tree from left to right, connecting each parent to max_children children until we run out of children to assign.
        # If there are enough children at each level, this will create a perfectly balanced tree (we assume we are in this case).
        # If not, the last parents at each level will have fewer children.
        for lvl in range(len(level_indices) - 1):
            # Collect all indices for the current level (parents) and the next level (children)
            parents = level_indices[lvl]
            children = level_indices[lvl + 1]

            child_ptr = 0
            for p in parents:
                # Connect each parent to max_children children, until we run out of children to assign
                for _ in range(self.max_children):
                    # If no more children to assign, stop
                    if child_ptr >= len(children):
                        break
                    c = children[child_ptr]
                    self.edges[p].append(c)
                    self.edges[c].append(p)
                    child_ptr += 1


    
    def _sanitize_transition_probs(self):
        """
        Convert word-based transition probabilities into
        index-based ones, keeping only valid tree edges.
        """
        cleaned = {}

        if not self.transition_probs:
            self.transition_probs = {}
            return

        # Check that all source and target words exist in the tree and that transitions are valid (i.e., between parent and child nodes).
        for src_word, targets in self.transition_probs.items():
            if src_word not in self.word_to_index:
                print(f"Warning: source word '{src_word}' not found in tree, skipping this probability.")
                continue

            src = self.word_to_index[src_word]
            neighbors = set(self.edges[src])

            valid = {}
            # Check all target words for the current source word, ensuring they exist in the tree and are valid transitions (i.e., parent-child relationships).
            for tgt_word, weight in targets.items():
                if tgt_word not in self.word_to_index:
                    print(f"Warning: target word '{tgt_word}' not found in tree, skipping this probability.")
                    continue

                tgt = self.word_to_index[tgt_word]
                if tgt not in neighbors:
                    print(f"ERROR: Invalid transition '{src_word}' -> '{tgt_word}' (not parent/child).")
                    continue

                if weight < 0:
                    print(f"ERROR: Negative weight for '{src_word}' -> '{tgt_word}'.")
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

        # generates a random number between 0 and total
        r = rng.random() * total
        acc = 0.0
        # Iterate through the neighbors and their weights, accumulating the weights until we exceed r, at which point we return the corresponding neighbor.
        # This effectively samples from the weighted distribution defined by the transition probabilities.
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
        elif not (0 <= start < len(self.words)):
            print(f"Invalid start index {start}") 
            start = rng.randrange(len(self.words))

        current = start
        seq = [self.words[current]]

        for _ in range(length - 1):
            current = self._choose_next(current, rng)
            if self.only_leaves:
                # If we want only leaves, we add node to sequence only if it's a leaf (i.e., has no children). If it's not a leaf, we keep traversing down until we find one.
                while len(self.edges[current])>1:
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
        transition_probs: Optional[Dict[Tuple[str, ...], Dict[Tuple[str, ...], float]]] = None,
        only_leaves = False
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
        self.only_leaves = only_leaves

        self.clusters: List[Tuple[str, ...]] = []
        self.edges: Dict[int, List[int]] = {}

        self._build_tree()
        self._sanitize_transition_probs()

    # -------------------------
    # Tree construction
    # -------------------------

    def _build_tree(self):
        # Flatten words and keep index mapping
        index = 0
        level_indices: List[List[int]] = []

        # Enumerate clusters through the tree. The root cluster will have idx 0, its children will have indices 1 to max_children, and so on.
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
        # This is the same logic as in WordTree, but we are connecting clusters instead of individual words.
        # The max_children parameter still applies to how many child clusters each parent cluster can have.
        # If there are enough clusters at each level, this will create a perfectly balanced tree.
        # If not, the last parents at each level will have fewer children.
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

                if weight < 0:
                    print(f"ERROR: Negative weight for {src_word} -> {tgt_word}.")
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
        elif not (0 <= start < len(self.clusters)):
            print(f"Invalid start index {start}")
            start = rng.randrange(len(self.clusters))

        current = start
        seq = [rng.choice(self.clusters[current])]

        for _ in range(length - 1):
            current = self._choose_next(current, rng)
            if self.only_leaves:
                while len(self.edges[current]) > 1:
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


class WordCircle:
    """
    Words arranged in a circular structure.

    Rules:
    - No custom transition probabilities.
    - Sampling happens by:
        1) Picking a random word from the circle.
        2) Moving left or right with equal probability.
        3) Producing a pair of tokens.
    - Output is formatted as:
        word1 word2
        word3 word4
        ...
    - Total number of tokens equals the requested length.
    """

    def __init__(self, words: List[str]):
        self.words = words
        self.size = len(words)

    # -------------------------
    # Sampling logic
    # -------------------------

    def generate_sequence(
        self,
        length: int,
        rng: Optional[random.Random] = None
    ) -> str:
        if self.size == 0 or length <= 0:
            return ""

        if rng is None:
            rng = random.Random(0)

        lines = []
        token_count = 0

        while token_count < length:
            # Step 1: sample random starting index
            start = rng.randrange(self.size)

            # Step 2: choose direction (-1 = left, +1 = right)
            direction = rng.choice([-1, 1])
            neighbor = (start + direction) % self.size

            w1 = self.words[start]
            w2 = self.words[neighbor]

            lines.append(f"{w1} {w2}")
            token_count += 2

        # If length is odd, trim the last extra token
        if token_count > length:
            last_line = lines[-1].split()
            lines[-1] = last_line[0]  # keep only first token

        return ", ".join(lines)

    # -------------------------
    # Debug / visualization
    # -------------------------

    def print_circle(self):
        print("Graph structure:")
        print("  ".join(self.words) + "\n")

    def save_circle(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("  ".join(self.words) + "\n")

if __name__ == "__main__":
    uncorr_words = [
            ["blackout",   "mafia",     "flu",     "lexical"],
            ["nonatomic",  "beverage",  "albums",  "crappy"],
            ["potassium",  "phoenix",   "grinder", "standby"],
            ["peanuts",    "undergrad", "culprit", "vitae"]
    ]
    wg1 = WordGrid(uncorr_words, torus=False)
    wg1.print_grid()

    levels = [
            ["blackout"],
            ["mafia", "flu"],
            ["lexical", "nonatomic", "beverage", "albums"],
            ["crappy","potassium", "phoenix", "grinder", "standby","peanuts", "undergrad", "culprit"],
            ["vitae","swagger", "tumult", "handful", "overwhelm", "subtitle","preserving", "plagiarism", "borrowers", "curled","embodiment", "interpol", "resizing", "oath", "defy", "certifications"]
        ]
    
    transition_probs = {
        "blackout": {"mafia": 0.5, "flu": 0.5},
        "mafia": {"blackout": 0.1, "lexical": 0.45, "nonatomic": 0.45},
        "flu": {"blackout": 0.1, "beverage": 0.45, "albums": 0.45},
        "lexical": {"mafia": 0.1, "crappy": 0.45, "potassium": 0.45},
        "nonatomic": {"mafia": 0.1, "phoenix": 0.45, "grinder": 0.45},
        "beverage": {"flu": 0.1, "standby": 0.45, "peanuts": 0.45},
        "albums": {"flu": 0.1, "undergrad": 0.45, "culprit": 0.45},
        "crappy": {"lexical": 0.1, "vitae": 0.45, "swagger": 0.45},
        "potassium": {"lexical": 0.1, "tumult": 0.45, "handful": 0.45},
        "phoenix": {"nonatomic": 0.1, "overwhelm": 0.45, "subtitle": 0.45},
        "grinder": {"nonatomic": 0.1, "preserving": 0.45, "plagiarism": 0.45},
        "standby": {"beverage": 0.1, "borrowers": 0.45, "curled": 0.45},
        "peanuts": {"beverage": 0.1, "embodiment": 0.45, "interpol": 0.45},
        "undergrad": {"albums": 0.1, "resizing": 0.45, "oath": 0.45},
        "culprit": {"albums": 0.1, "defy": 0.45, "certifications": 0.45}
    }
    tree = WordTree(levels, max_children=2, transition_probs=transition_probs)
    tree.print_tree()

    levels = [
        [("blackout", "vitae","swagger")],
        [("mafia","tumult", "handful"), ("flu","overwhelm","subtitle")],
        [("lexical","preserving", "plagiarism"), ("nonatomic","borrowers", "curled"), ("beverage","embodiment", "interpol"), ("albums","resizing", "oath")],
        [("crappy","defy","certifications"),("potassium", "albeit", "mote"), ("phoenix", "tasty", "wealthiest"), ("grinder", "unconditional", "intends"), ("standby", "flaming", "fabs"),("peanuts", "stricter", "improvised"), ("undergrad", "soar", "finns"), ("culprit", "righteous", "intimately")]
    ]

    transition_probs = {
        ("blackout", "vitae","swagger"): {
            ("mafia","tumult", "handful"): 0.25,
            ("flu","overwhelm","subtitle"): 0.75
        },
        ("mafia","tumult", "handful"): {
            ("blackout", "vitae","swagger"): 0,
            ("lexical","preserving", "plagiarism"): 0.25,
            ("nonatomic","borrowers", "curled"): 0.75
        },
        ("flu","overwhelm","subtitle"): {
            ("blackout", "vitae","swagger"): 0,
            ("beverage","embodiment", "interpol"): 0.25,
            ("albums","resizing", "oath"): 0.75
        },
        ("lexical","preserving", "plagiarism"): {
            ("mafia","tumult", "handful"): 0,
            ("crappy","defy","certifications"): 0.25,
            ("potassium", "albeit", "mote"): 0.75
        },
        ("nonatomic","borrowers", "curled"): {
            ("mafia","tumult", "handful"): 0,
            ("phoenix", "tasty", "wealthiest"): 0.25,
            ("grinder", "unconditional", "intends"): 0.75
        },
        ("beverage","embodiment", "interpol"): {
            ("flu","overwhelm","subtitle"): 0,
            ("standby", "flaming", "fabs"): 0.25,
            ("peanuts", "stricter", "improvised"): 0.75
        },
        ("albums","resizing", "oath"): {
            ("flu","overwhelm","subtitle"): 0,
            ("undergrad", "soar", "finns"): 0.25,
            ("culprit", "righteous", "intimately"): 0.75
        }
    }
    tree_cluster = WordTreeCluster(levels, max_children=2, transition_probs=transition_probs)
    tree_cluster.print_tree()

    circle_words = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    wc = WordCircle(circle_words)
    wc.print_circle()

    rng = random.Random(42)
    print(wc.generate_sequence(rng=rng, length=10))