from pathlib import Path
from structures import WordTree, WordTreeCluster
from structures import WordGrid


BASE_DIR = Path(__file__).resolve().parent.parent  # Project/
DATA_DIR = BASE_DIR / "data"


def generate_dataset_multiple_rw(structure, context_length: int, sequence_length: int, output_path: str):
    '''
    Function to generate a dataset based on the provided structure. It saves the generated data in the directory Data in the output path.
    
    :param structure: Object describing the dataset structure
    :param context_length: Number of examples to generate
    :type context_length: int
    :param sequence_length: Length of each sequence
    :type sequence_length: int
    :param output_path: Path to save the generated dataset
    :type output_path: str
    '''
    output_path = DATA_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = ","
    with open(f"{output_path}", "w") as f:
        for _ in range(context_length-1):
            sequence = structure.generate_sequence(sequence_length)        
            f.write(sep.join(sequence) + "\n")
        final_query = structure.generate_sequence(sequence_length-1)
        f.write(sep.join(final_query) + f"{sep}?")


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
    # Define common parameters for context window (they use around 600 tokens)
    context_size = 600

    # First experiment: Grid structure (8x8) with probability distribution over transitions
    # Read from selected_llama31_layer0.txt the words: there is one word per each line and a number that i have to ignore
    # I read 8 words and put them in a list, the list of lists is the word object that is given to the grid

    input_file = BASE_DIR / "uncorrelated-words" / "selected_llama31_layer0.txt"
    with open(input_file, "r") as f:
        lines = f.readlines()
        words = []
        for i in range(0, 64, 8):
            row = []
            for j in range(8):
                word = lines[i+j].split()[0]  # Get the first part of the line (the word)
                row.append(word.strip())  # Remove any leading/trailing whitespace
            words.append(row)
    wg1 = WordGrid(words, torus=False)
    wg1.print_grid()
    generate_dataset(wg1, context_tokens=context_size, output_path=f"grid_dataset_{context_size}.txt")

    # Second experiment: Torus structure (4x4)
    wg2 = WordGrid(words, torus=True)
    wg2.print_grid()
    generate_dataset(wg2, context_tokens=context_size, output_path=f"torus_dataset_{context_size}.txt")

    # Third experiment: Binary Tree structure - height 3
    # content_size = 60 # Update context size because the number of sequences is smaller
    # sequence_size = 10 # Update sequence size because the number of nodes is smaller
    # levels = [
    #     ["grape"],
    #     ["lamp", "birch"],
    #     ["eye", "bishop", "blue", "sprinkler"]
    # ]
    # tree = WordTree(levels, max_children=2)
    # tree.print_tree()
    # generate_dataset(tree, context_tokens=context_size, output_path=f"bin_tree_dataset_{context_size}.txt")

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