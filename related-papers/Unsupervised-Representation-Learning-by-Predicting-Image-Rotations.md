# Unsupervised Representation Learning by Predicting Image Rotations

## Bullet Point Summary

- **Task:** Any visual task
  - But tests run are (mostly) Image Classification
- **Why:** Achieve good results using unlabeled data
  - Beacuse data is cheaper
  - a bit clickbait: **semi**supervised would have been more correct
    - Pretrain to predict image rotations using convnets
      - Data automatically labeled in input
    - Freeze (some?) convolutional layers, re-train fully connected layers for needed tasks
      - Here labels are needed
    - Basically transfer learning (?)
- **How:** Apply a rotation, ask network to guess applied rotation
  - Rotations leave no artifacts, so the model can't use them to predict rotations
  - If it learns how to recognize rotations, it means it is (also) learning high level object semantics
    - Attention maps seems to prove this right
  - The rotations **is** the label
    - A label of $y$ means applying a clock-wise rotations of $90°$ for $y-1$ times
      - e.g.: $y=2$ means an image rotated by $(y-1)\cdot90°=1\cdot90°=90°$
      - $y=0$ is not used for loss functions reasons ($\log(y)$)
- **Many settings were tried**
  - Instead of $0,90,180,270$ degrees rotations use only a subset of them
    - e.g. only rotate by $0$ or $180$ degrees
    - Performed worse
  - Istead of rotating by $90°$ multiples rotate by $45°$ multiples
    - Performed worse
  - Use more/less convolutional layers
    - More layers seems to be better, but after some layers performance drops
- **Results Comparison*