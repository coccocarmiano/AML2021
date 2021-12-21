# Domain Generalization by Solving Jigsaw Puzzles
## Introduction
- Objective: Improve object (image) classification across different domains (Domain Generalization - Domain Adaptation)
- Method: **JiGen** - Unsupervised learning: performs jointly image classification and Jigsaw puzzle solving to improve regularization and generalization
  ![JiGen Concept](/related-papers/resources/JiGen_concept.png "JiGen Concept")

## Method: JiGen
- Setting:
  - Multiple source domains (labeled)
  - Need to perform well on any target domain (unlabeled)
- Approach:
  - Reuse any network able to do image classification (good)
  - Put an extra head (last layer) with a fully connected layer to perform JigSaw Puzzle reconstruction
  - Optimize an objective made by a weighted sum: *ClassificationLoss + JigSaw-Puzzle-Reconstruction-Loss*
  - Split the source images in NxN patches (N is fixed, usual = 3)
  - Generate P permutations of the patches (usual P=30) according to Hamming-distance and assign an index to each permutation;
  - The reconstruction is a simple classification task in which the network has to predict the correct index (the original index) starting from a permutation of patches (another index)
- Why it works? It improves regularization and generalization while learning JigSaw puzzle reconstruction (unsupervised task)

  ![JiGen Schema](/related-papers/resources/JiGen_schema.png "JiGen Schema")
