# Halva

[![PyPI - Version](https://img.shields.io/pypi/v/halva)](https://pypi.org/project/halva/)

*Multivariate analysis of ordinal data with missing values and latent variables*

Halva---‘grapHical Analysis with Latent VAriables’---is a Python package dedicated to statistical analysis of multivariate ordinal data, designed specifically to handle missing values and latent variables in a similar way to the full information maximum likelihood (FIML) method.

Instead of assuming that the ordinal data comes approximately from a multivariate normal distribution ([which can systematically lead to errors](https://doi.org/10.1016/j.jesp.2018.08.009)), Halva uses a rigorous statistical model adapted to ordinal data (namely, a multivariate ordered probit model).

## Installation

Halva can be installed using [pip](https://pypi.org/project/halva/):

```bash
pip install halva
```

## Basic usage

```python
import pandas as pd
import halva

# Load data
data = pd.read_excel('my_data')

# Option: add structure constraints
edge_list = [(0, 1), (1, 2), (2, 3)]

# Perform inference (likelihood maximization)
res = halva.infer_precision(data, edges=edge_list)

# Show precision matrix
print(res.theta)
```
