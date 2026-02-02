
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch


# Download and infer on model Llama-3.2-1B

# For each of the dataset files do:

    # Pass to LLama the prompt inside the dataset file

    # Download the hidden representation at various layers

    # Visualize the hidden representation using PCA (with 2/3 principal components): tokens representing the same words should be colored the same

    # Save results

# Plot everything