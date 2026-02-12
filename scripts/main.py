
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch


# Download and infer on model Llama3.1-8B

# For each of the dataset files do:

    # Pass to LLama the dataset file as the prompt

    # Download the hidden representation of the last 50 tokens at various layers: FOCUS ON LAYER 26 AS THEY SAW IT GIVES THE BEST RESULTS

    # Visualize the hidden representation using PCA (with 2/3 principal components): tokens representing the same words should be coloured the same

    # Save results

# Plot everything