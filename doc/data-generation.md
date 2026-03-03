# Dataset generation
We first started by generating the context files. We have three possible structures from which we can sample: grid, circle and tree. The first two structure were also present in the original paper while we added the third one.

We debated a lot on how to construct the context: if it was better to use one long random walk or multiple short ones. 
In the end, for the grid and tree structures, we constructed the context with a single line representing one random walk on the graph. We deemed this method more simple and, since we still obtained good result, quite effective. 

First, we tried to arrange both numbers and days in a circle. Note that, as it was shown in literature, the internal representations of the days of the week are naturally arranged in circles.

Circle with numbers ordered increasingly:
1  2  3  4  5  6  7  8  9  10  11  12  13  14  15

Circle with numbers shuffled:
9  14  8  7  15  13  6  3  10  4  5  12  1  2  11

Circle with days ordered:
Monday  Tuesday  Wednesday  Thursday  Friday  Saturday  Sunday

Circle with days shuffled:
Tuesday  Thursday  Friday  Wednesday  Sunday  Monday  Saturday

Of course, we realized that for numbers a helix would have been better suited, though we still believed that the ordered circle would be easier to learn than the shuffled one.

We sampled from the circle in a similar way as they did in the paper: we randomly sampled pairs of adjacent nodes and separated each pair from others with a comma.

Example for ordered circle of days:
"Sunday Monday, Sunday Monday, Monday Tuesday, Friday Saturday, ..."

This way to extract the representations of the concepts we just needed to ignore the whitespace and comma tokens.

We then tried to arrange the number in a grid. We tried both a "ordered" one and a shuffled one.

Grid with numbers ordered increasingly:
1   2   3   4   5 
6   7   8   9   10
11  12  13  14  15

Grid with numbers in random order:
9   14  8   7   15
13  6   3   10  4 
5   12  1   2   11

With seven days there was no way to construct a grid so we skipped that structure.

We sampled from the grid by starting from a random node and performing a random walk across the edges of the graph. From each node there is a uniform probability of moving in any of the connected nodes. As they did in the paper, we also observed that the numbers in the middle of the grid were sampled sligthly more by construction. We separated each number with a white space.

Example for grid of ordered numbers:
"9 4 3 4 3 2 1 2 ..."

Finally, we arranged numbers in three different trees, two "ordered" ones and one completely deprived of meaningful disposition. We also tried to arrange days of the week in a tree, though since it was hard to define a meaningful order we just sampled from a random tree

Tree where each left branch corresponds to a mathematical subtraction and each right branch corresponds to a summation:
8
├── 4
│   ├── 2
│   │   ├── 1
│   │   └── 3
│   └── 6
│       ├── 5
│       └── 7
└── 12
    ├── 10
    │   ├── 9
    │   └── 11
    └── 14
        ├── 13
        └── 15

Tree where the leaves are ordered increasingly:
A
├── B
│   ├── D
│   │   ├── 1
│   │   └── 2
│   └── E
│       ├── 3
│       └── 4
└── C
    ├── F
    │   ├── 5
    │   └── 6
    └── G
        ├── 7
        └── 8

Tree with numbers in random order:
3
├── 13
│   ├── 14
│   │   ├── 12
│   │   └── 5
│   └── 15
│       ├── 2
│       └── 6
└── 1
    ├── 11
    │   ├── 9
    │   └── 8
    └── 10
        ├── 4
        └── 7

Tree with days in random order:
Wednesday
├── Sunday
│   ├── Thursday
│   └── Friday
└── Tuesday
    ├── Saturday
    └── Monday

We can sample from trees in two different ways: either by traversing them up and down multiple times and recording each node visited or by traversing them up and down multiple times and recording only the leaf nodes visited. For the tree with ordered leaves, we sample only in the second way. Either way, the starting node is sampled randomly across the tree and at each node the probability of going to any of the connected nodes is uniform among them.

Example for tree with numbers in random order:
"4 10 4 10 4 10 4 10 4 10 7 ..."