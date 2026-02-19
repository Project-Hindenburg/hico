from pathlib import Path
import random
import numpy as np
from structures import WordTree, WordTreeCluster
from structures import WordGrid


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


if __name__ == "__main__":
    context_sizes = [300,600,1200,1800]
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
        
        batch_size = len(words) * len(words[0])  # Total number of words in the grid
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        # Generate batch dataset
        wg0.save_grid(f"{DATA_DIR}/multi_batch/paper_grid/paper_grid_structure.txt")
        generate_dataset_multiple_rw(wg0, batch_size=batch_size, sequence_length=sequence_length, output_path=f"multi_batch/paper_grid/paper_grid_{context_size}.txt")

        # Generate single rw dataset
        wg0.save_grid(f"{DATA_DIR}/one_random_walk/paper_grid/paper_grid_structure.txt")
        generate_dataset(wg0, context_tokens=context_size, output_path=f"one_random_walk/paper_grid/paper_grid_one_rw_{context_size}.txt")


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

        
        batch_size = len(uncorr_words) * len(uncorr_words[0])  # Total number of words in the grid
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        # batch dataset
        wg1.save_grid(f"{DATA_DIR}/multi_batch/grid_16/grid_dataset_structure.txt")
        generate_dataset_multiple_rw(wg1, batch_size=batch_size, sequence_length=sequence_length, output_path=f"{DATA_DIR}/multi_batch/grid_16/grid_dataset_{context_size}.txt")

        # single rw dataset
        wg1.save_grid(f"{DATA_DIR}/one_random_walk/grid_16/grid_dataset_structure.txt")
        generate_dataset(wg1, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/grid_16/grid_dataset_one_rw_{context_size}.txt")

        # Grid (4x4) experiment with uncorrelated words and custom probability transitions ----------------------------------------------------------------------------
        print("-" * 80)
        print(f"Grid experiment with custom probabilities for context size: {context_size}")

        # Define custom transition probabilities for each word in the grid
        transition_probabilities = {
            "blackout": {"mafia": 0.5, "nonatomic": 0.5},
            "mafia": {"blackout": 0.4, "beverage": 0.2, "flu": 0.4},
            "flu": {"mafia": 0.4, "lexical": 0.4, "albums": 0.2},
            "lexical": {"flu": 0.5, "crappy": 0.5},
            "nonatomic": {"blackout": 0.4, "beverage": 0.2, "potassium": 0.4},
            "beverage": {"mafia": 0.1, "nonatomic": 0.1, "albums": 0.4, "phoenix": 0.4},
            "albums": {"flu": 0.1, "beverage": 0.4, "crappy": 0.1, "grinder": 0.4},
            "crappy": {"lexical": 0.4, "albums": 0.2, "standby": 0.4},
            "potassium": {"nonatomic": 0.4, "phoenix": 0.2, "peanuts": 0.4},
            "phoenix": {"beverage": 0.4, "potassium": 0.1, "grinder": 0.4, "undergrad": 0.1},
            "grinder": {"albums": 0.4, "phoenix": 0.4, "standby": 0.1, "culprit": 0.1},
            "standby": {"crappy": 0.4, "grinder": 0.2, "vitae": 0.4},
            "peanuts": {"potassium": 0.5, "undergrad": 0.5},
            "undergrad": {"phoenix": 0.2, "peanuts": 0.4, "culprit": 0.4},
            "culprit": {"grinder": 0.2, "undergrad": 0.4, "vitae": 0.4},
            "vitae": {"standby": 0.5, "culprit": 0.5}
        }        

        wgprob = WordGrid(uncorr_words, torus=False, transition_probs=transition_probabilities)
        wgprob.print_grid()

        batch_size = len(uncorr_words) * len(uncorr_words[0])  # Total number of words in the grid
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        # batch dataset
        wgprob.save_grid(f"{DATA_DIR}/multi_batch/parametrized_grid_16/grid_dataset_structure.txt")
        generate_dataset_multiple_rw(wgprob, batch_size=batch_size, sequence_length=sequence_length, output_path=f"{DATA_DIR}/multi_batch/parametrized_grid_16/grid_dataset_{context_size}.txt")

        # single rw dataset
        wgprob.save_grid(f"{DATA_DIR}/one_random_walk/parametrized_grid_16/grid_dataset_structure.txt")
        generate_dataset(wgprob, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/parametrized_grid_16/grid_dataset_one_rw_{context_size}.txt")

        # Torus (4x4) experiment with uncorrelated words ----------------------------------------------------------------------------
        print("-" * 80)
        print(f"Torus experiment for context size: {context_size}")
        wg2 = WordGrid(uncorr_words, torus=True)
        wg2.print_grid()

        batch_size = len(uncorr_words) * len(uncorr_words[0])  # Total number of words in the grid
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        #batch dataset
        wg2.save_grid(f"{DATA_DIR}/multi_batch/torus_16/torus_dataset_structure.txt")
        generate_dataset_multiple_rw(wg2, batch_size=batch_size, sequence_length=sequence_length, output_path=f"{DATA_DIR}/multi_batch/torus_16/torus_dataset_{context_size}.txt")

        #single rw dataset
        wg2.save_grid(f"{DATA_DIR}/one_random_walk/torus_16/torus_dataset_structure.txt")
        generate_dataset(wg2, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/torus_16/torus_dataset_one_rw_{context_size}.txt")

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

        batch_size = sum(len(level) for level in levels)  # Number of nodes in the tree
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        # batch dataset
        tree.save_tree(f"{DATA_DIR}/multi_batch/tree_4_levels/bin_tree_structure.txt")
        generate_dataset_multiple_rw(tree, batch_size=batch_size, sequence_length=sequence_length, output_path=f"{DATA_DIR}/multi_batch/tree_4_levels/bin_tree_{context_size}.txt")

        # single rw dataset
        tree.save_tree(f"{DATA_DIR}/one_random_walk/tree_4_levels/bin_tree_structure.txt")
        generate_dataset(tree, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/tree_4_levels/bin_tree_one_rw_{context_size}.txt")

        # Binary tree (height 2) experiment with uncorrelated words and custom probability transitions ----------------------------------------------------------
        print("-" * 80)
        print(f"Binary tree with custom probabilities experiment for context size: {context_size}")
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

        tree_prob = WordTree(levels, max_children=2, transition_probs=transition_probs)
        tree_prob.print_tree()

        batch_size = sum(len(level) for level in levels)  # Number of nodes in the tree
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        # batch dataset
        tree_prob.save_tree(f"{DATA_DIR}/multi_batch/parametrized_tree_4_levels/bin_tree_structure.txt")
        generate_dataset_multiple_rw(tree_prob, batch_size=batch_size, sequence_length=sequence_length, output_path=f"{DATA_DIR}/multi_batch/parametrized_tree_4_levels/bin_tree_{context_size}.txt")
        
        # single rw dataset
        tree_prob.save_tree(f"{DATA_DIR}/one_random_walk/parametrized_tree_4_levels/bin_tree_structure.txt")
        generate_dataset(tree_prob, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/parametrized_tree_4_levels/bin_tree_one_rw_{context_size}.txt")

        # Binary tree (height 2) experiment with days of the week ------------------------------------------------------------ 
        print("-" * 80)
        print(f"Binary tree with days of the week experiment for context size: {context_size}") 
        levels = [
            ["Wednesday"],
            ["Sunday", "Friday"],
            ["Thursday", "Friday", "Saturday", "Monday"]
        ]
        tree = WordTree(levels, max_children=2)
        tree.print_tree()
        generate_dataset(tree, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/tree_days/bin_tree_days_dataset_{context_size}.txt")

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

        batch_size = sum(len(level) for level in levels)  # Number of nodes in the tree
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        # batch dataset
        tree_cluster.save_tree(f"{DATA_DIR}/multi_batch/tree_clusters_3_levels/bin_tree_cluster_structure.txt")
        generate_dataset_multiple_rw(tree_cluster, batch_size=batch_size, sequence_length=sequence_length, output_path=f"{DATA_DIR}/multi_batch/tree_clusters_3_levels/bin_tree_cluster_{context_size}.txt")

        # single rw dataset
        tree_cluster.save_tree(f"{DATA_DIR}/one_random_walk/tree_clusters_3_levels/bin_tree_cluster_structure.txt")
        generate_dataset(tree_cluster, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/tree_clusters_3_levels/bin_tree_cluster_one_rw_{context_size}.txt")

        # Binary tree (height 3) experiment with clusters of words and custom probability transitions ----------------------------------------------------------
        print("-" * 80)
        print(f"Binary tree with clusters of words and custom probabilities experiment for context size: {context_size}")
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
        tree_cluster_prob = WordTreeCluster(levels, max_children=2, transition_probs=transition_probs)
        tree_cluster_prob.print_tree()

        batch_size = sum(len(level) for level in levels)  # Number of nodes in the tree
        sequence_length = context_size  # Length of each sequence to generate
        print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")

        # batch dataset
        tree_cluster_prob.save_tree(f"{DATA_DIR}/multi_batch/parametrized_tree_clusters_3_levels/bin_tree_cluster_structure.txt")
        generate_dataset_multiple_rw(tree_cluster_prob, batch_size=batch_size, sequence_length=sequence_length, output_path=f"{DATA_DIR}/multi_batch/parametrized_tree_clusters_3_levels/bin_tree_cluster_{context_size}.txt")

        # single rw dataset
        tree_cluster_prob.save_tree(f"{DATA_DIR}/one_random_walk/parametrized_tree_clusters_3_levels/bin_tree_cluster_structure.txt")
        generate_dataset(tree_cluster_prob, context_tokens=context_size, output_path=f"{DATA_DIR}/one_random_walk/parametrized_tree_clusters_3_levels/bin_tree_cluster_one_rw_{context_size}.txt")

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
