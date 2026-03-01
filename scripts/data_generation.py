from pathlib import Path
import random
import numpy as np
from structures import WordTree, WordCircle
from structures import WordGrid
from typing import List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent  # Project/
DATA_DIR = BASE_DIR / "data"


def generate_dataset_multiple_rw(structure, batch_size: int, sequence_length: int, output_path: str, seed: int = 0):
    '''
    Function to generate a dataset based on the provided structure. It saves the generated data in the directory Data in the output path.
    
    :param structure: Object describing the dataset structure
    :param batch_size: Number of sequences to generate in each random walk
    :type batch_num: int
    :param batch_size: Number of sequences to generate in each random walk
    :type batch_size: int
    :param sequence_length: Length of each sequence
    :type sequence_length: int
    :param output_path: Path to save the generated dataset
    :type output_path: str
    :param seed: Random seed for reproducibility
    :type seed: int
    '''
    output_path = DATA_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = " "
    rng = random.Random(seed)
    with open(f"{output_path}", "w") as f:
        perm = np.random.RandomState(seed=seed).permutation(batch_size)
        for i in range(batch_size):
            sequence = structure.generate_sequence(sequence_length, start=int(perm[i]), rng = rng)        
            f.write(sep.join(sequence) + "\n")

def generate_dataset(structure, context_tokens: int, output_path: str, seed: int = 0):
    '''
    Function to generate a dataset based on the provided structure. It saves the generated data in the directory Data in the output path.
    
    :param structure: Object describing the dataset structure
    :param context_tokens: Number of tokens in the context
    :type context_tokens: int
    :param output_path: Path to save the generated dataset
    :type output_path: str
    '''
    output_path = DATA_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = " "
    rng = random.Random(seed)
    with open(f"{output_path}", "w") as f:
        sequence = structure.generate_sequence(context_tokens, rng=rng)   
        # if sequence is a string, write it directly, otherwise join with separator
        if isinstance(sequence, str):
            f.write(sequence)
        else:
            f.write(sep.join(sequence))

def shuffle_tree_levels(
    levels: List[List[str]],
    rng: Optional[random.Random] = None
) -> List[List[str]]:
    """
    Takes a tree-level list like:

        [
            ["root"],
            ["a", "b"],
            ["c", "d", "e", "f"],
            ...
        ]

    Returns a new levels list with:
    - Same structure (same number of levels and same counts per level)
    - Same words
    - Words randomly redistributed across levels/positions
    """

    if rng is None:
        rng = random.Random()

    # Flatten all words
    all_words = [word for level in levels for word in level]

    # Shuffle them
    rng.shuffle(all_words)

    # Rebuild levels with original structure
    new_levels = []
    idx = 0
    for level in levels:
        size = len(level)
        new_levels.append(all_words[idx:idx + size])
        idx += size

    return new_levels

def shuffle_grid(
    grid: List[List[str]],
    rng: Optional[random.Random] = None
) -> List[List[str]]:
    """
    Takes a 2D grid like:

        [
            ["1", "2", "3"],
            ["4", "5", "6"]
        ]

    Returns a new grid with:
    - Same shape
    - Same words
    - Words shuffled across positions
    """

    if rng is None:
        rng = random.Random()

    # Flatten grid
    all_words = [word for row in grid for word in row]

    # Shuffle
    rng.shuffle(all_words)

    # Rebuild grid with original shape
    new_grid = []
    idx = 0
    for row in grid:
        row_len = len(row)
        new_grid.append(all_words[idx:idx + row_len])
        idx += row_len

    return new_grid

if __name__ == "__main__":
    # Number tree experiment ---------------------------------------------------------- 
    numbers = [
        ["8"],
        ["4", "12"],
        ["2", "6", "10", "14"],
        ["1", "3", "5", "7", "9", "11", "13", "15"]
        ]
    tree = WordTree(levels = numbers, max_children=2)
    tree.print_tree()
    tree.save_tree(f"{DATA_DIR}/one_random_walk/tree_numbers/structure.txt")
    generate_dataset(tree, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/tree_numbers/dataset.txt")

    # Shuffled numbers tree experiment ----------------------------------------------------------
    shuffled_numbers = shuffle_tree_levels(numbers, rng=random.Random(42))
    shuffled_tree = WordTree(levels = shuffled_numbers, max_children=2)
    shuffled_tree.print_tree()
    shuffled_tree.save_tree(f"{DATA_DIR}/one_random_walk/shuffled_tree_numbers/structure.txt")
    generate_dataset(shuffled_tree, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/shuffled_tree_numbers/dataset.txt")

    # Only leaves tree experiment ----------------------------------------------------------
    only_leaves_tree = WordTree(levels = numbers, max_children=2, only_leaves=True)
    only_leaves_tree.print_tree()
    only_leaves_tree.save_tree(f"{DATA_DIR}/one_random_walk/only_leaves_tree/structure.txt")
    generate_dataset(only_leaves_tree, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/only_leaves_tree/dataset.txt")

    # Only leaves shuffled tree experiment ----------------------------------------------------------
    only_leaves_shuffled_tree = WordTree(levels = shuffled_numbers, max_children=2, only_leaves=True)
    only_leaves_shuffled_tree.print_tree()
    only_leaves_shuffled_tree.save_tree(f"{DATA_DIR}/one_random_walk/only_leaves_shuffled_tree/structure.txt")
    generate_dataset(only_leaves_shuffled_tree, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/only_leaves_shuffled_tree/dataset.txt")

    # Only Leaves ordered tree experiment ----------------------------------------------------------
    only_leaves = [
        ["A"],
        ["B", "C"],
        ["D", "E", "F", "G"],
        ["1", "2", "3", "4", "5", "6", "7", "8"]
    ]
    only_leaves_ordered_tree = WordTree(levels = only_leaves, max_children=2, only_leaves=True)
    only_leaves_ordered_tree.print_tree()
    only_leaves_ordered_tree.save_tree(f"{DATA_DIR}/one_random_walk/only_leaves_ordered_tree/structure.txt")
    generate_dataset(only_leaves_ordered_tree, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/only_leaves_ordered_tree/dataset.txt")

    # Numbers grid experiment ----------------------------------------------------------

    numbers_grid = [
        ["1", "2", "3", "4", "5"],
        ["6", "7", "8", "9", "10"],
        ["11", "12", "13", "14", "15"]
    ]
    grid = WordGrid(grid=numbers_grid)
    grid.print_grid()
    grid.save_grid(f"{DATA_DIR}/one_random_walk/grid_numbers/structure.txt")
    generate_dataset(grid, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/grid_numbers/dataset.txt")

    # Number grid square experiment ----------------------------------------------------------
    numbers_grid_square = [
        ["1", "2", "3", "4"],
        ["5", "6", "7", "8"],
        ["9", "10", "11", "12"],
        ["13", "14", "15", "16"]
    ]
    grid_square = WordGrid(grid=numbers_grid_square)
    grid_square.print_grid()
    grid_square.save_grid(f"{DATA_DIR}/one_random_walk/grid_square_numbers/structure.txt")
    generate_dataset(grid_square, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/grid_square_numbers/dataset.txt")

    # Shuffled numbers grid square experiment ----------------------------------------------------------
    shuffled_numbers_square = shuffle_grid(numbers_grid_square, rng=random.Random(42))
    shuffled_grid_square = WordGrid(grid=shuffled_numbers_square)
    shuffled_grid_square.print_grid()
    shuffled_grid_square.save_grid(f"{DATA_DIR}/one_random_walk/shuffled_grid_square_numbers/structure.txt")
    generate_dataset(shuffled_grid_square, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/shuffled_grid_square_numbers/dataset.txt")

    # Shuffled numbers grid experiment ----------------------------------------------------------
    shuffled_numbers = shuffle_grid(numbers_grid, rng=random.Random(42))
    shuffled_grid = WordGrid(grid=shuffled_numbers)
    shuffled_grid.print_grid()
    shuffled_grid.save_grid(f"{DATA_DIR}/one_random_walk/shuffled_grid_numbers/structure.txt")
    generate_dataset(shuffled_grid, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/shuffled_grid_numbers/dataset.txt")

    # Numbers circle experiment ----------------------------------------------------------

    numbers_circle = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    circle = WordCircle(words=numbers_circle)
    circle.print_circle()
    circle.save_circle(f"{DATA_DIR}/one_random_walk/circle_numbers/structure.txt")
    generate_dataset(circle, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/circle_numbers/dataset.txt")

    # Numbers circle experiment alternative sampling ----------------------------------------------------------
    circle_alt = WordCircle(words=numbers_circle, alternative_sampling=True)
    circle_alt.print_circle()
    circle_alt.save_circle(f"{DATA_DIR}/one_random_walk/circle_numbers_alt/structure.txt")
    generate_dataset(circle_alt, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/circle_numbers_alt/dataset.txt")
    
    # Shuffled numbers circle experiment ----------------------------------------------------------
    rng_circle = random.Random(42)
    shuffled_numbers_circle = numbers_circle.copy()
    rng_circle.shuffle(shuffled_numbers_circle)

    shuffled_circle = WordCircle(words=shuffled_numbers_circle)
    shuffled_circle.print_circle()
    shuffled_circle.save_circle(f"{DATA_DIR}/one_random_walk/shuffled_circle_numbers/structure.txt")
    generate_dataset(shuffled_circle, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/shuffled_circle_numbers/dataset.txt")

    # Shuffled numbers circle experiment alternative sampling ----------------------------------------------------------
    shuffled_circle_alt = WordCircle(words=shuffled_numbers_circle, alternative_sampling=True)
    shuffled_circle_alt.print_circle()
    shuffled_circle_alt.save_circle(f"{DATA_DIR}/one_random_walk/shuffled_circle_numbers_alt/structure.txt")
    generate_dataset(shuffled_circle_alt, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/shuffled_circle_numbers_alt/dataset.txt")

    # Days of the week tree experiment ----------------------------------------------------------
    levels = [
        ["Wednesday"],
        ["Sunday", "Tuesday"],
        ["Thursday", "Friday", "Saturday", "Monday"]
    ]
    tree = WordTree(levels, max_children=2)
    tree.print_tree()
    tree.save_tree(f"{DATA_DIR}/one_random_walk/tree_days/structure.txt")
    generate_dataset(tree, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/tree_days/dataset.txt")

    # Days of the week circle experiment ----------------------------------------------------------
    days_circle = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    circle = WordCircle(words=days_circle)
    circle.print_circle()
    circle.save_circle(f"{DATA_DIR}/one_random_walk/circle_days/structure.txt")
    generate_dataset(circle, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/circle_days/dataset.txt")

    # Days of the week circle experiment alternative sampling ----------------------------------------------------------
    circle_alt = WordCircle(words=days_circle, alternative_sampling=True)
    circle_alt.print_circle()
    circle_alt.save_circle(f"{DATA_DIR}/one_random_walk/circle_days_alt/structure.txt")
    generate_dataset(circle_alt, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/circle_days_alt/dataset.txt")

    # Shuffled days of the week circle experiment ----------------------------------------------------------
    rng_circle = random.Random(42)
    shuffled_days_circle = days_circle.copy()
    rng_circle.shuffle(shuffled_days_circle)

    shuffled_circle = WordCircle(words=shuffled_days_circle)
    shuffled_circle.print_circle()
    shuffled_circle.save_circle(f"{DATA_DIR}/one_random_walk/shuffled_circle_days/structure.txt")
    generate_dataset(shuffled_circle, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/shuffled_circle_days/dataset.txt")

    # Shuffled days of the week circle experiment alternative sampling ----------------------------------------------------------
    shuffled_circle_alt = WordCircle(words=shuffled_days_circle, alternative_sampling=True)
    shuffled_circle_alt.print_circle()
    shuffled_circle_alt.save_circle(f"{DATA_DIR}/one_random_walk/shuffled_circle_days_alt/structure.txt")
    generate_dataset(shuffled_circle_alt, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/shuffled_circle_days_alt/dataset.txt")

    # Days of the week grid experiment ----------------------------------------------------------
    days_grid = [
        ["Sunday", "Monday", "Tuesday"],
        ["Wednesday", "Thursday", "Friday"],
    ]
    grid = WordGrid(grid=days_grid)
    grid.print_grid()
    grid.save_grid(f"{DATA_DIR}/one_random_walk/grid_days/structure.txt")
    generate_dataset(grid, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/grid_days/dataset.txt")

    # Uncorrelated words tree experiment ----------------------------------------------------------
    numbers = [
        ["blackout"],
        ["mafia", "flu"],
        ["lexical", "nonatomic", "beverage", "albums"],
        ["crappy","potassium", "phoenix", "grinder", "standby","peanuts", "undergrad", "culprit"],
    ]
    tree = WordTree(levels = numbers, max_children=2)
    tree.print_tree()
    tree.save_tree(f"{DATA_DIR}/one_random_walk/tree_uncorrelated/structure.txt")
    generate_dataset(tree, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/tree_uncorrelated/dataset.txt")

    # Uncorrelated words grid experiment ----------------------------------------------------------
    numbers_grid = [
        ["blackout",   "mafia",     "flu",     "lexical","nonatomic"],
        ["beverage",  "albums",  "crappy","potassium",  "phoenix"],
        ["grinder", "standby","peanuts",    "undergrad", "culprit"]
    ]
    grid = WordGrid(grid=numbers_grid)
    grid.print_grid()
    grid.save_grid(f"{DATA_DIR}/one_random_walk/grid_uncorrelated/structure.txt")
    generate_dataset(grid, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/grid_uncorrelated/dataset.txt")

    # Uncorrelated words circle experiment ----------------------------------------------------------
    numbers_circle = ["blackout", "mafia", "flu", "lexical","nonatomic", "beverage", "albums", "crappy","potassium", "phoenix", "grinder", "standby","peanuts", "undergrad", "culprit"]
    circle = WordCircle(words=numbers_circle)
    circle.print_circle()
    circle.save_circle(f"{DATA_DIR}/one_random_walk/circle_uncorrelated/structure.txt")
    generate_dataset(circle, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/circle_uncorrelated/dataset.txt")

    # Uncorrelated words circle experiment alternative sampling ----------------------------------------------------------
    circle_alt = WordCircle(words=numbers_circle, alternative_sampling=True)
    circle_alt.print_circle()
    circle_alt.save_circle(f"{DATA_DIR}/one_random_walk/circle_uncorrelated_alt/structure.txt")
    generate_dataset(circle_alt, context_tokens=2000, output_path=f"{DATA_DIR}/one_random_walk/circle_uncorrelated_alt/dataset.txt")
