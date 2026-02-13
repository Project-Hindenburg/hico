from pathlib import Path
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
    with open(f"{output_path}", "w") as f:
        perm = np.random.RandomState(seed=seed).permutation(batch_size)
        for i in range(batch_size):
            sequence = structure.generate_sequence(sequence_length, start=int(perm[i]))        
            f.write(sep.join(sequence) + "\n")

def generate_dataset(structure, context_tokens: int, output_path: str):
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
    with open(f"{output_path}", "w") as f:
        sequence = structure.generate_sequence(context_tokens)   
        f.write(sep.join(sequence))


if __name__ == "__main__":
    context_sizes = [300,600,1200,1800]
    for context_size in context_sizes:
        # Paper grid experiment ---------------------------------------------------------------------------------------------
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

        # Grid (4x4) experiment with uncorrelated words ----------------------------------------------------------------------------

        uncorr_words = [
                ["blackout", "mafia", "flu", "lexical"],
                ["nonatomic", "beverage", "albums", "crappy"],
                ["potassium", "phoenix", "grinder", "standby"],
                ["peanuts", "undergrad", "culprit", "vitae"]
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

        # Torus (4x4) experiment with uncorrelated words ----------------------------------------------------------------------------
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
        levels = [
            ["blackout"],
            ["mafia", "flu"],
            ["lexical", "nonatomic", "beverage", "albums"],
            ["crappy","potassium", "phoenix", "grinder", "standby","peanuts", "undergrad", "culprit"],
            ["vitae","swagger", "tumult", "handful", "overwhelm", "subtitle","preserving", "plagiarism", "borrowers", "curled","embodiment", "interpol", "resizing", "oath", "defy", "certifications"]
        ]
        tree = WordTree(levels, max_children=2)
        tree.print_tree()



        #generate_dataset(tree, context_tokens=context_size, output_path=f"bin_tree_dataset_{context_size}.txt")

        # Fourth experiment: binary tree structure with days of the week - height 3
        # levels = [
        #     ["Wednesday"],
        #     ["Sunday", "Friday"],
        #     ["Thursday", "Friday", "Saturday", "Monday"]
        # ]
        # tree = WordTree(levels, max_children=2)
        # tree.print_tree()
        # generate_dataset(tree, context_tokens=context_size, output_path=f"bin_tree_days_dataset_{context_size}.txt")

        # Fifth experiment: binary tree structure with clusters of words - height 3
        # c1 = ("grape", "apple")
        # c2 = ("lamp", "lantern")
        # c3 = ("container", "box")
        # c4 = ("eye", "ear")
        # c5 = ("bishop", "knight")
        # c6 = ("school", "university")
        # c7 = ("sprinkler", "hose")
        # cluster_levels = [[c1], [c2, c3], [c4, c5, c6, c7]]
        # tree_cluster = WordTreeCluster(cluster_levels, max_children=2)
        # tree_cluster.print_tree()
        # generate_dataset(tree_cluster, context_tokens=context_size, output_path=f"bin_tree_cluster_dataset_{context_size}.txt")