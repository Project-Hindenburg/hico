from pathlib import Path
from structures import WordTree
from structures import WordGrid


BASE_DIR = Path(__file__).resolve().parent.parent  # Project/
DATA_DIR = BASE_DIR / "data"


def generate_dataset(structure, context_length: int, sequence_length: int, output_path: str):
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
        for _ in range(context_length):
            sequence = structure.generate_sequence(sequence_length)        
            f.write(sep.join(sequence) + "\n")
        final_query = structure.generate_sequence(sequence_length-1)
        f.write(sep.join(final_query) + f"{sep}?\n")

if __name__ == "__main__":
    # Define common parameters fo context window
    context_size = 30
    sequence_size = 15

    # First experiment: Grid structure with probability distribution over transitions
    words =[
            ["sand", "handle", "math"],
            ["queue", "biscuit", "straw"],
            ["birch", "shampoo", "trumpet"]
        ]
    
    probabilities = {
        "sand": {"handle": 0.5, "queue": 0.5},
        "handle": {"sand": 0.33, "math": 0.33, "biscuit": 0.34},
        "math": {"handle": 0.5, "straw": 0.5},
        "queue": {"sand": 0.33, "biscuit": 0.33, "birch": 0.34},
        "biscuit": {"handle": 0.25, "queue": 0.25, "straw": 0.25, "shampoo": 0.25},
        "straw": {"math": 0.33, "biscuit": 0.33, "trumpet": 0.34},
        "birch": {"queue": 0.5, "shampoo": 0.5},
        "shampoo": {"biscuit": 0.33, "birch": 0.33, "trumpet": 0.34},
        "trumpet": {"straw": 0.5, "shampoo": 0.5}
    }
    wg1 = WordGrid(words, torus=False, transition_probs=probabilities)
    wg1.print_grid()
    generate_dataset(wg1, context_length=context_size, sequence_length=sequence_size, output_path="grid_with_probs_dataset.txt")

    # Second experiment: Torus structure
    wg2 = WordGrid(words, torus=True)
    wg2.print_grid()
    generate_dataset(wg2, context_length=context_size, sequence_length=sequence_size, output_path="torus_dataset.txt")

    # Third experiment: Binary Tree structure - height 3
    sequence_size = 10 # Update sequence size because the number of nodes is smaller
    levels = [
        ["grape"],
        ["lamp", "birch"],
        ["eye", "bishop", "school", "sprinkler"]
    ]
    tree = WordTree(levels, max_children=2)
    tree.print_tree()
    generate_dataset(tree, context_length=context_size, sequence_length=sequence_size, output_path="bin_tree_dataset.txt")
