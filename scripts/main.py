
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch


# Download and infer on model Llama-3.2-1B

# For each of the dataset files do:

    # Pass to LLama the dataset file as the prompt

    # Download the hidden representation at various layers: FOCUS ON LAYER 26 AS THEY SAW IT GIVES THE BEST RESULTS

    # Visualize the hidden representation using PCA (with 2/3 principal components): tokens representing the same words should be coloured the same

    # Save results

# Plot everything