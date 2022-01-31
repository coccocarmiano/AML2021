# Results

|   S   |    T    | Epochs |   LR   | $\alpha_1$ | $\alpha_2$ |  MH   |  CL   | $\lambda$ |  AUC   |  HOS  | HOS*  |
| :---: | :-----: | :----: | :----: | :--------: | :--------: | :---: | :---: | :-------: | :----: | :---: | :---: |
|  Art  | Clipart | 30+30  |  0.01  |    3.5     |    2.5     |  On   |  On   |   0.01    | 0.5556 | 69.79 | 73.84 |
|  Art  | Clipart | 30+30  |  0.01  |    3.5     |    2.5     |  On   |  Off  |     -     | 0.5659 | 55.66 | 61.01 |
|  Art  | Clipart | 30+30  |  0.01  |    3.5     |    2.5     |  Off  |  Off  |     -     | 0.5038 | 32.54 | 49.67 |
|  Art  | Clipart | 20+20  | 0.0005 |    3.0     |    2.0     |  Off  |  On   |   0.01    | 0.5140 | 58.13 | 64.14 |


# Problems

Row 3: AUC is too low because model overfitted, must re-run

Generic: maybe we should use a better scheduler

# Guidelines for Parameters

In Step One, the task is minimizing the rotation classification error, using the classification task as a regularizer. Hence, $\alpha_1$ and $\lambda$ should be picked so to lower consistently and "fast enough" the `Rotatation Loss` and the `Rotation Loss`, while also lowering, but not necessarily as much or as fast the `Classification Loss`. This is because we are trying to pre-train the model ( the `feature_extractor`, ad least ) to extract meaningful features from data.

In Step Two, the process should be more balanced. The $\lambda$ value could be left the same given that the `Center Loss` value is generically lower than the `Rotation Loss` one and that both combined are the regularizer in this phase, but to make sure that the actual task being trained is (open set) classification the $\alpha_2$ value should be reduced by 30-70% the value of $\alpha_1$ more or less.