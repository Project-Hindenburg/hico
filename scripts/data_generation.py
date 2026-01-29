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
    with open(f"{output_path}", "w") as f:
        for _ in range(context_length):
            sequence = structure.generate_sequence(sequence_length)        
            f.write(" ".join(sequence) + "\n")

if __name__ == "__main__":
    wg1 = WordGrid(
        [
        ["clock", "evaporator", "logic"],
        ["queue", "biscuit", "straw"],
        ["tree", "shampoo", "tarpaulin"]
        ],
        torus=False
    )
    wg2 = WordGrid(
        [
        ["clock", "evaporator", "logic"],
        ["queue", "biscuit", "straw"],
        ["tree", "shampoo", "tarpaulin"]
        ],
        torus=True
    )
    generate_dataset(wg1, context_length=5, sequence_length=10, output_path="grid_dataset.txt")
    generate_dataset(wg2, context_length=5, sequence_length=10, output_path="torus_dataset.txt")

    levels = [
        ["grape"],
        ["lamp", "container"],
        ["eye", "bishop", "school", "sprinkler"]
    ]

    tree = WordTree(levels, max_children=2)
    generate_dataset(tree, context_length=5, sequence_length=10, output_path="bin_tree_dataset.txt")
