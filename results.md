
| **Source** | **Target** | **Epochs** | **Multi-Head** | $\alpha_1$ | **Center-Loss** | **AUC ROC** |
| :--------: | :--------: | :--------: | :------------: | :--------: | :-------------: | :---------: |
|  Clipart   |  Product   |     20     |      True      |    3.0     |       Off       |   0.5757    |
| RealWorld  |  Product   |     20     |      True      |    3.0     |       Off       |   0.5964    |
|  Product   |    Art     |     20     |      True      |    3.0     |       Off       |   0.5453    |
|  Clipart   |    Art     |     20     |      True      |    3.0     |       Off       |   0.5315    |
|  Clipart   |    Art     |     30     |      True      |    3.0     |       Off       |   0.5470    |
|  Clipart   |    Art     |     30     |      True      |    3.0     |       1.0       |   0.5223    |
|    Art     |  Clipart   |     30     |      True      |    4.0     |       Off       |   0.5604    |
|    Art     |  Clipart   |     10     |      True      |    4.0     |       Off       |   0.5440    |
|    Art     |  Clipart   |     10     |      True      |    4.0     |  0.01 / 0.001   |   0.5626    |
|    Art     |  Clipart   |     30     |      True      |    4.0     |  0.01 / 0.001   |   0.5774    |