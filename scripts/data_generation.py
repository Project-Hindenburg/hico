import random
from typing import List, Tuple, Dict

class WordGrid:
    def __init__(self, grid: List[List[str]], pacman: bool = False):
        """
        Initialize with a 2D grid of words.

        pacman:
            False -> normal boundaries
            True  -> wrap-around (Pac-Man style)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.pacman = pacman

        self.adjacency = self._build_adjacency()
        self.longest_word_length = max(
            len(word) for row in grid for word in row
        )

    def _build_adjacency(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Build adjacency graph using horizontal & vertical neighbors."""
        adj = {}

        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = []

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc

                    if self.pacman:
                        # wrap around edges
                        nr %= self.rows
                        nc %= self.cols
                        neighbors.append((nr, nc))
                    else:
                        # normal boundary
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            neighbors.append((nr, nc))

                adj[(r, c)] = neighbors

        return adj

    def generate_sequence(
        self,
        length: int,
        start: Tuple[int, int] = None
    ) -> List[str]:
        """Generate a random walk of words."""
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
            current = random.choice(self.adjacency[current])
            r, c = current
            sequence.append(self.grid[r][c])

        return sequence

    def print_grid(self):
        """Pretty-print the grid."""
        for row in self.grid:
            print("  ".join(f"{word:<{self.longest_word_length}}" for word in row))


if __name__ == "__main__":
    grid = [
        ["sun", "apple", "logic"],
        ["car", "chair", "love"],
        ["tree", "book", "tarpaulin"]
    ]

    print("Normal boundaries:")
    wg1 = WordGrid(grid, pacman=False)
    wg1.print_grid()
    print(" -> ".join(wg1.generate_sequence(8)))

    print("\nPac-Man boundaries:")
    wg2 = WordGrid(grid, pacman=True)
    wg2.print_grid()
    print(" -> ".join(wg2.generate_sequence(8)))
