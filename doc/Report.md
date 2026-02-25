# Hierarchical In Context Organizations

Recent work [1] shows that with enough demonstrations, LLMs can reorga-
nize concept representations to reflect context-specified structure (e.g., grid/ring
graphs), measurable via decreasing Dirichlet energy and increasing rule-following
accuracy, i.e., sum of probability mass assigned to valid continuations of se-
quences.

Specifically, we want to answer the following questions: What is the relation-
ship between phrase structures and distances in the representation space? [Apparently, no clear relation] Are
we able to detect hierarchical geometries induced in-context by simple trees or
grammars? [If this is the dendogram, yes we recover it] Lastly, how much the learned geometries are memory dependent? [We dropped this experiment]

The new main objective of this project is to study the connection between semantic and structure in the embeddings.
The main questions are "does the semantic have a specific role?" and "does the graph type have a specific role?" or equivalently "is one structure better suited for some concepts than the others?".

To answer these questions, we decided to perform the following experiments: arrange ordered numbers in a tree, arrange shuffled numbers in a tree, arrange ordered numbers in a grid, arrange shuffled numbers in a grid,  arrange ordered numbers in a circle, arrange shuffled numbers in a circle.

We sample from the tree by performing random walks that start from the root and traverse the tree up and down. Alternatively, we also perform the same random walks but saving only leaf nodes.

## References
1. C. F. Park, et al.; Iclr: In-context learning of representations. The Thirteenth International Conference on Learning Representations. 2024.
