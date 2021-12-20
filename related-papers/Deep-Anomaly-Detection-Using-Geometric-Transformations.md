## Problem: anomaly/novelty detection
Construct a classifier capable of detecting out-of-distribution samples, "abnormal" in respect to a "normal" class of samples.
One of the first approach to this problem using NNs.

## Proposed Approach
NN that discriminate between many types of geometric transformations applied to "normal" images. This lead to learn features useful for detecting anomalies. (Self-labeled dataset)

## Dataset
Expand initial train set appling all decided transformations to each sample and self label it.
One vs all approach: 1 class is the "normal" one and all the others are "abnormal" --> The single class is the one on which we have to learn a normality score.

## Network
Use train set (JUST NORMAL CLASS) to train a deep k-class classification model that classify which transformation is been applied. (cross-entropy loss)
Used a Wide-Residual-Net (WRN), for other hyperparams check *Hyperparameters and Optimization Methods* paper section.

## Normality Score
Scoring function to map test sample to a value (score) that represent how much the image is "normal".

Compute a Normality Score Ns over target (test) set. Ns is a value that shows how much "normal" is the analized test sample vs the main class, the higher Ns the "normal", the lower the further from the normal class.
The Dirichlet Normality Score function is used (check paper [2]) based on softmax output transformations.
We then need to define a threshold to decide at which score to separate unknown and known samples. --> The choice of this parameter is not addressed in this paper. --> Interesting approach used in paper [4], where the average Ns over the target domain is used as threshold to separate unkn/kn samples without the need to takle the problem.

To measure the performance in anomalies detection the area under receiver operating characteristic curve (AUROC) over Ns scores is used. If 0.5 it means that it's equal to a random decision at each sample. When prior knowledge on portion of anomalies is available we can consider the area under precision-recall curve (AUPR) as more accurate.

## Results
The Ns is applied to all test set (both normal and anomalies) and the AUROC was computed using available groud truth. State-of-the-art results at the time with big margin over previous works. 

## Plus
They also tried it a little bit with multiclass datasets out-of-distribution samples identification with double-head network (simil open-set problem): multi-class classification and transformation classification, the first one used only during training.

## Applied transformations
72 applied: horizontal flipping, translations, and rotations. Degraded performances with non-geometric ones.