# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/cooper-org/cooper/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/cooper/\_\_init\_\_.py                                           |       12 |        4 |     67% |     16-21 |
| src/cooper/cmp.py                                                    |      145 |        0 |    100% |           |
| src/cooper/constraints/\_\_init\_\_.py                               |        2 |        0 |    100% |           |
| src/cooper/constraints/constraint.py                                 |       41 |        1 |     98% |        58 |
| src/cooper/constraints/constraint\_state.py                          |       29 |        0 |    100% |           |
| src/cooper/formulations/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| src/cooper/formulations/formulations.py                              |       90 |        2 |     98% |    55, 57 |
| src/cooper/formulations/utils.py                                     |       31 |        1 |     97% |       164 |
| src/cooper/multipliers/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| src/cooper/multipliers/multipliers.py                                |       63 |        1 |     98% |       189 |
| src/cooper/optim/\_\_init\_\_.py                                     |        4 |        0 |    100% |           |
| src/cooper/optim/constrained\_optimizers/\_\_init\_\_.py             |        5 |        0 |    100% |           |
| src/cooper/optim/constrained\_optimizers/alternating\_optimizer.py   |       40 |        1 |     98% |       220 |
| src/cooper/optim/constrained\_optimizers/constrained\_optimizer.py   |       29 |        0 |    100% |           |
| src/cooper/optim/constrained\_optimizers/extrapolation\_optimizer.py |       38 |        1 |     97% |       171 |
| src/cooper/optim/constrained\_optimizers/simultaneous\_optimizer.py  |       16 |        1 |     94% |        55 |
| src/cooper/optim/optimizer.py                                        |       49 |        0 |    100% |           |
| src/cooper/optim/torch\_optimizers/\_\_init\_\_.py                   |        2 |        0 |    100% |           |
| src/cooper/optim/torch\_optimizers/extragradient.py                  |      123 |       36 |     71% |85, 96, 100-101, 109, 170, 172, 174, 185, 189-191, 201, 206, 208-215, 250, 252, 254, 256, 268-270, 274, 277, 279, 293, 297, 303, 310-312 |
| src/cooper/optim/torch\_optimizers/nupi\_optimizer.py                |      160 |       28 |     82% |134, 136, 138, 141, 149, 151, 153, 155, 192-193, 198, 215-222, 255, 284, 312-313, 346-347, 355, 365, 389, 420-421 |
| src/cooper/optim/unconstrained\_optimizer.py                         |       14 |        1 |     93% |        40 |
| src/cooper/penalty\_coefficients/\_\_init\_\_.py                     |        2 |        0 |    100% |           |
| src/cooper/penalty\_coefficients/penalty\_coefficient\_updaters.py   |       62 |        1 |     98% |        48 |
| src/cooper/penalty\_coefficients/penalty\_coefficients.py            |       54 |        0 |    100% |           |
| src/cooper/utils/\_\_init\_\_.py                                     |        2 |        0 |    100% |           |
| src/cooper/utils/annotations.py                                      |        8 |        0 |    100% |           |
| src/cooper/utils/utils.py                                            |        7 |        0 |    100% |           |
| testing/\_\_init\_\_.py                                              |        2 |        0 |    100% |           |
| testing/cooper\_helpers.py                                           |      137 |        2 |     99% |  267, 298 |
| testing/utils.py                                                     |       45 |       12 |     73% |36, 41, 52, 55, 58, 61-63, 67, 70, 74, 77 |
| tests/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| tests/conftest.py                                                    |        7 |        0 |    100% |           |
| tests/constraints/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| tests/constraints/test\_constraint.py                                |       58 |        0 |    100% |           |
| tests/constraints/test\_constraint\_state.py                         |       75 |        0 |    100% |           |
| tests/formulations/\_\_init\_\_.py                                   |        0 |        0 |    100% |           |
| tests/formulations/test\_formulation.py                              |       72 |        0 |    100% |           |
| tests/formulations/test\_formulation\_utils.py                       |       65 |        0 |    100% |           |
| tests/multipliers/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| tests/multipliers/conftest.py                                        |       26 |        0 |    100% |           |
| tests/multipliers/test\_explicit\_multipliers.py                     |       87 |        0 |    100% |           |
| tests/multipliers/test\_implicit\_multiplier.py                      |        7 |        0 |    100% |           |
| tests/optim/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| tests/optim/conftest.py                                              |       14 |        0 |    100% |           |
| tests/optim/test\_constrained\_optimizers.py                         |       28 |        0 |    100% |           |
| tests/optim/test\_optimizer.py                                       |       37 |        0 |    100% |           |
| tests/optim/torch\_optimizers/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| tests/optim/torch\_optimizers/test\_nupi.py                          |      193 |        1 |     99% |        40 |
| tests/penalty\_coefficients/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| tests/penalty\_coefficients/test\_penalty\_coefficients.py           |       52 |        0 |    100% |           |
| tests/penalty\_coefficients/test\_penalty\_updater.py                |       94 |        0 |    100% |           |
| tests/pipeline/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| tests/pipeline/conftest.py                                           |      132 |        1 |     99% |        43 |
| tests/pipeline/test\_checkpoint.py                                   |       68 |        0 |    100% |           |
| tests/pipeline/test\_convergence.py                                  |       34 |        0 |    100% |           |
| tests/pipeline/test\_manual.py                                       |      170 |        1 |     99% |       299 |
| tests/test\_cmp.py                                                   |      225 |        1 |     99% |        68 |
|                                                            **TOTAL** | **2658** |   **96** | **96%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/cooper-org/cooper/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/cooper-org/cooper/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/cooper-org/cooper/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/cooper-org/cooper/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fcooper-org%2Fcooper%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/cooper-org/cooper/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.