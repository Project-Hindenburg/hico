from pathlib import Path
import random
import numpy as np
from structures import WordTree, WordTreeCluster
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

if __name__ == "__main__":
    context_sizes = [500,1000,1500,2000,2500]
    for context_size in context_sizes:
        # Paper grid experiment ---------------------------------------------------------------------------------------------
        print("-" * 80)
        print(f"Paper experiment for context size: {context_size}")
        words = [
                ["apple", "bird", "car", "egg"],
                ["house", "milk", "plane", "opera"],
                ["box", "sand", "sun", "mango"],
                ["rock", "math", "code", "phone"]
            ]
        wg0 = WordGrid(words, torus=False)
        wg0.print_grid()
        
        sequence_length = context_size  # Length of each sequence to generate

        # Generate single rw dataset
        wg0.save_grid(f"{DATA_DIR}/one_random_walk/paper_grid/paper_grid_structure.txt")
        #generate_dataset(wg0, context_tokens=context_size, output_path=f"one_random_walk/paper_grid/paper_grid_one_rw_{context_size}.txt")


        # Grid (4x4) experiment with uncorrelated words ----------------------------------------------------------------------------
        print("-" * 80)
        print(f"Grid experiment for context size: {context_size}")

        uncorr_words = [
                ["blackout",   "mafia",     "flu",     "lexical"],
                ["nonatomic",  "beverage",  "albums",  "crappy"],
                ["potassium",  "phoenix",   "grinder", "standby"],
                ["peanuts",    "undergrad", "culprit", "vitae"]
        ]
        wg1 = WordGrid(uncorr_words, torus=False)
        wg1.print_grid()
        sequence_length = context_size  # Length of each sequence to generate
        # single rw dataset
        wg1.save_grid(f"{DATA_DIR}/one_random_walk/grid_16/grid_dataset_structure.txt")
        #generate_dataset(wg1, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/grid_16/grid_dataset_one_rw_{context_size}.txt")

        # Grid (4x4) experiment with uncorrelated words and custom probability transitions ----------------------------------------------------------------------------
        print("-" * 80)
        print(f"Grid experiment with custom probabilities for context size: {context_size}")

        # Define custom transition probabilities for each word in the grid
        transition_probabilities = {
            # --- Corners ---
            "blackout": {"mafia": 0.5, "nonatomic": 0.5},
            "lexical": {"flu": 0.5, "crappy": 0.5},
            "peanuts": {"potassium": 0.5, "undergrad": 0.5},
            "vitae": {"standby": 0.5, "culprit": 0.5},
            # --- Edges ---
            "mafia": {"blackout": 0.45, "beverage": 0.2, "flu": 0.35},
            "flu": {"mafia": 0.35, "lexical": 0.45, "albums": 0.2},
            "nonatomic": {"blackout": 0.45, "beverage": 0.2, "potassium": 0.35},
            "crappy": {"lexical": 0.45, "albums": 0.2, "standby": 0.35},
            "potassium": {"nonatomic": 0.35, "phoenix": 0.2, "peanuts": 0.45},
            "standby": {"crappy": 0.35, "grinder": 0.2, "vitae": 0.45},
            "undergrad": {"phoenix": 0.2, "peanuts": 0.45, "culprit": 0.35},
            "culprit": {"grinder": 0.2, "undergrad": 0.35, "vitae": 0.45},
            # --- Interior ---
            "beverage": {"mafia": 0.25, "nonatomic": 0.25, "albums": 0.25, "phoenix": 0.25},
            "albums": {"flu": 0.25, "beverage": 0.25, "crappy": 0.25, "grinder": 0.25},
            "phoenix": {"beverage": 0.25, "potassium": 0.25, "grinder": 0.25, "undergrad": 0.25},
            "grinder": {"albums": 0.25, "phoenix": 0.25, "standby": 0.25, "culprit": 0.25},
        }
        wgprob = WordGrid(uncorr_words, torus=False, transition_probs=transition_probabilities)
        wgprob.print_grid()
        sequence_length = context_size  # Length of each sequence to generate
        # single rw dataset
        wgprob.save_grid(f"{DATA_DIR}/one_random_walk/parametrized_grid_16/grid_dataset_structure.txt")
        #generate_dataset(wgprob, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/parametrized_grid_16/grid_dataset_one_rw_{context_size}.txt")

        # Binary tree (height 4) experiment with uncorrelated words ----------------------------------------------------------
        print("-" * 80)
        print(f"Binary tree experiment for context size: {context_size}")
        levels = [
            ["blackout"],
            ["mafia", "flu"],
            ["lexical", "nonatomic", "beverage", "albums"],
            ["crappy","potassium", "phoenix", "grinder", "standby","peanuts", "undergrad", "culprit"],
            ["vitae","swagger", "tumult", "handful", "overwhelm", "subtitle","preserving", "plagiarism", "borrowers", "curled","embodiment", "interpol", "resizing", "oath", "defy", "certifications"]
        ]
        tree = WordTree(levels, max_children=2)
        tree.print_tree()
        sequence_length = context_size  # Length of each sequence to generate

        new_levels = shuffle_tree_levels(levels)
        tree_shuffled = WordTree(new_levels, max_children=2)
        tree_shuffled.print_tree()
        # single rw dataset
        tree.save_tree(f"{DATA_DIR}/one_random_walk/tree_4_levels/bin_tree_structure.txt")
        #generate_dataset(tree, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/tree_4_levels/bin_tree_one_rw_{context_size}.txt")

        # Binary tree (height 2) experiment with uncorrelated words and custom probability transitions ----------------------------------------------------------
        print("-" * 80)
        print(f"Binary tree with custom probabilities experiment for context size: {context_size}")
        transition_probs = {
        "blackout": {"mafia": 0.4, "flu": 0.6},

        "mafia": {"blackout": 0.2, "lexical": 0.24, "nonatomic": 0.56},
        "flu": {"blackout": 0.2, "beverage": 0.24, "albums": 0.56},
        
        "lexical": {"mafia": 0.2, "crappy": 0.16, "potassium": 0.64},
        "nonatomic": {"mafia": 0.2, "phoenix": 0.16, "grinder": 0.64},
        "beverage": {"flu": 0.2, "standby": 0.16, "peanuts": 0.64},
        "albums": {"flu": 0.2, "undergrad": 0.16, "culprit": 0.64},
        
        "crappy": {"lexical": 0.2, "vitae": 0.08, "swagger": 0.72},
        "potassium": {"lexical": 0.2, "tumult": 0.08, "handful": 0.72},
        "phoenix": {"nonatomic": 0.2, "overwhelm": 0.08, "subtitle": 0.72},
        "grinder": {"nonatomic": 0.2, "preserving": 0.08, "plagiarism": 0.72},
        "standby": {"beverage": 0.2, "borrowers": 0.08, "curled": 0.72},
        "peanuts": {"beverage": 0.2, "embodiment": 0.08, "interpol": 0.72},
        "undergrad": {"albums": 0.2, "resizing": 0.08, "oath": 0.72},
        "culprit": {"albums": 0.2, "defy": 0.08, "certifications": 0.72}
        }
        tree_prob = WordTree(levels, max_children=2, transition_probs=transition_probs)
        tree_prob.print_tree()
        sequence_length = context_size  # Length of each sequence to generate
        # single rw dataset
        tree_prob.save_tree(f"{DATA_DIR}/one_random_walk/parametrized_tree_4_levels/bin_tree_structure.txt")
        #generate_dataset(tree_prob, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/parametrized_tree_4_levels/bin_tree_one_rw_{context_size}.txt")

        # Binary tree (height 2) experiment with uncorrelated words and only leaves ----------------------------------------------------------
        print("-" * 80)
        print(f"Binary tree with only leaves experiment for context size: {context_size}")
        tree_leaves = WordTree(levels, max_children=2, only_leaves=True)
        tree_leaves.print_tree()
        sequence_length = context_size  # Length of each sequence to generate
        # single rw dataset
        tree_leaves.save_tree(f"{DATA_DIR}/one_random_walk/only_leaves_tree_4_levels/bin_tree_structure.txt")
        #generate_dataset(tree_leaves, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/only_leaves_tree_4_levels/bin_tree_one_rw_{context_size}.txt")

        # Binary tree (height 2) experiment with days of the week ------------------------------------------------------------ 
        print("-" * 80)
        print(f"Binary tree with days of the week experiment for context size: {context_size}") 
        levels = [
            ["Wednesday"],
            ["Sunday", "Tuesday"],
            ["Thursday", "Friday", "Saturday", "Monday"]
        ]
        tree = WordTree(levels, max_children=2)
        tree.print_tree()
        tree.save_tree(f"{DATA_DIR}/one_random_walk/tree_days/bin_tree_structure.txt")
        #generate_dataset(tree, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/tree_days/bin_tree_days_dataset_{context_size}.txt")

        # Binary tree (height 3) experiment with clusters of words ----------------------------------------------------------
        print("-" * 80)
        print(f"Binary tree with clusters of words experiment for context size: {context_size}")
        levels = [
            [("blackout", "vitae","swagger")],
            [("mafia","tumult", "handful"), ("flu","overwhelm","subtitle")],
            [("lexical","preserving", "plagiarism"), ("nonatomic","borrowers", "curled"), ("beverage","embodiment", "interpol"), ("albums","resizing", "oath")],
            [("crappy","defy","certifications"),("potassium", "albeit", "mote"), ("phoenix", "tasty", "wealthiest"), ("grinder", "unconditional", "intends"), ("standby", "flaming", "fabs"),("peanuts", "stricter", "improvised"), ("undergrad", "soar", "finns"), ("culprit", "righteous", "intimately")]
        ]
        tree_cluster = WordTreeCluster(levels, max_children=2)
        tree_cluster.print_tree()
        sequence_length = context_size  # Length of each sequence to generate
        # single rw dataset
        tree_cluster.save_tree(f"{DATA_DIR}/one_random_walk/tree_clusters_3_levels/bin_tree_cluster_structure.txt")
        #generate_dataset(tree_cluster, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/tree_clusters_3_levels/bin_tree_cluster_one_rw_{context_size}.txt")

        # Binary tree (height 3) experiment with clusters of words and custom probability transitions ----------------------------------------------------------
        print("-" * 80)
        print(f"Binary tree with clusters of words and custom probabilities experiment for context size: {context_size}")
        transition_probs = {
            ("blackout", "vitae","swagger"): {
                ("mafia","tumult", "handful"): 0.4,
                ("flu","overwhelm","subtitle"): 0.6
            },

            ("mafia","tumult", "handful"): {
                ("blackout", "vitae","swagger"): 0.2,
                ("lexical","preserving", "plagiarism"): 0.24,
                ("nonatomic","borrowers", "curled"): 0.56
            },
            ("flu","overwhelm","subtitle"): {
                ("blackout", "vitae","swagger"): 0.2,
                ("beverage","embodiment", "interpol"): 0.24,
                ("albums","resizing", "oath"): 0.56
            },

            ("lexical","preserving", "plagiarism"): {
                ("mafia","tumult", "handful"): 0.2,
                ("crappy","defy","certifications"): 0.16,
                ("potassium", "albeit", "mote"): 0.64
            },
            ("nonatomic","borrowers", "curled"): {
                ("mafia","tumult", "handful"): 0.2,
                ("phoenix", "tasty", "wealthiest"): 0.16,
                ("grinder", "unconditional", "intends"): 0.64
            },
            ("beverage","embodiment", "interpol"): {
                ("flu","overwhelm","subtitle"): 0.2,
                ("standby", "flaming", "fabs"): 0.16,
                ("peanuts", "stricter", "improvised"): 0.64,
            },
            ("albums","resizing", "oath"): {
                ("flu","overwhelm","subtitle"): 0.2,
                ("undergrad", "soar", "finns"): 0.16,
                ("culprit", "righteous", "intimately"): 0.64
            }
        }
        tree_cluster_prob = WordTreeCluster(levels, max_children=2, transition_probs=transition_probs)
        tree_cluster_prob.print_tree()
        sequence_length = context_size  # Length of each sequence to generate
        # single rw dataset
        tree_cluster_prob.save_tree(f"{DATA_DIR}/one_random_walk/parametrized_tree_clusters_3_levels/bin_tree_cluster_structure.txt")
        #generate_dataset(tree_cluster_prob, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/parametrized_tree_clusters_3_levels/bin_tree_cluster_one_rw_{context_size}.txt")

        # Grid (8x8) experiment with uncorrelated words ---------------------------------------------------------------------------------------------
        # input_file = BASE_DIR / "uncorrelated-words" / "selected_llama31_layer0.txt"
        # with open(input_file, "r") as f:
        #     lines = f.readlines()
        #     words = []
        #     for i in range(0, 64, 8):
        #         row = []
        #         for j in range(8):
        #             word = lines[i+j].split()[0]  # Get the first part of the line (the word)
        #             row.append(word.strip())  # Remove any leading/trailing whitespace
        #         words.append(row)
