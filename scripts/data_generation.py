from pathlib import Path
from structures import WordTree
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

    # First experiment: Grid structure (4x4) with probability distribution over transitions
    words =[
            ["sand",  "handle",  "math",      "grape" ],
            ["queue", "biscuit", "straw",     "lamp"  ],
            ["birch", "shampoo", "trumpet",   "school"],
            ["quilt", "bishop",  "sprinkler", "bee"   ]
        ]
    
    probabilities = {
        # Row 0
        "sand":    {"handle": 0.5, "queue": 0.5},
        "handle":  {"sand": 0.33, "math": 0.33, "biscuit": 0.34},
        "math":    {"handle": 0.33, "grape": 0.33, "straw": 0.34},
        "grape":   {"math": 0.5, "lamp": 0.5},

        # Row 1
        "queue":   {"sand": 0.33, "biscuit": 0.34, "birch": 0.33},
        "biscuit": {"queue": 0.25, "handle": 0.25, "straw": 0.25, "shampoo": 0.25},
        "straw":   {"biscuit": 0.25, "math": 0.25, "lamp": 0.25, "trumpet": 0.25},
        "lamp":    {"straw": 0.33, "grape": 0.33, "school": 0.34},

        # Row 2
        "birch":   {"queue": 0.33, "shampoo": 0.34, "quilt": 0.33},
        "shampoo": {"birch": 0.25, "biscuit": 0.25, "trumpet": 0.25, "bishop": 0.25},
        "trumpet": {"shampoo": 0.25, "straw": 0.25, "school": 0.25, "sprinkler": 0.25},
        "school":  {"trumpet": 0.33, "lamp": 0.33, "bee": 0.34},

        # Row 3
        "quilt":      {"birch": 0.5, "bishop": 0.5},
        "bishop":     {"quilt": 0.33, "shampoo": 0.33, "sprinkler": 0.34},
        "sprinkler":  {"bishop": 0.33, "trumpet": 0.33, "bee": 0.34},
        "bee":        {"sprinkler": 0.5, "school": 0.5},
    }

    wg1 = WordGrid(words, torus=False, transition_probs=probabilities)
    wg1.print_grid()
    generate_dataset(wg1, context_tokens=context_size, output_path="grid_with_probs_dataset.txt")

    # Second experiment: Torus structure (4x4)
    wg2 = WordGrid(words, torus=True)
    wg2.print_grid()
    generate_dataset(wg2, context_tokens=context_size, output_path="torus_dataset.txt")

    # Third experiment: Binary Tree structure - height 3
    content_size = 60 # Update context size because the number of sequences is smaller
    sequence_size = 10 # Update sequence size because the number of nodes is smaller
    levels = [
        ["grape"],
        ["lamp", "birch"],
        ["eye", "bishop", "school", "sprinkler"]
    ]
    tree = WordTree(levels, max_children=2)
    tree.print_tree()
    generate_dataset(tree, context_tokens=context_size, output_path="bin_tree_dataset.txt")
