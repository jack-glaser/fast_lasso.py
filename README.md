# fast_lasso.py
[LASSO regression](https://en.wikipedia.org/wiki/Lasso_(statistics)) is a popular tool for addressing the variable selection problem in statistical inference. This package provides a streamlined and quantitatively optimized implementation of the non-stochastic cyclical coordinate descent solution to LASSO and iteratively implements the model over decreasing values of the main parameter <img src="https://latex.codecogs.com/gif.latex?\lambda" /> . 

# Computational Motivation
Implementation of LASSO is often paired with ridge regression (and/or [elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization), the hybridized form). Since stochastic gradient descent provides asymptotically converging solutions in most cases across all these functional forms, it is often implemented as the computational solution for any choice elastic net parameters (including the LASSO case, where <img src="https://latex.codecogs.com/gif.latex?\lambda_2" /> = 0). 

While these programs can be helpful for researchers who want to include objective function selection as part of their analysis, LASSO is often preferred due to its coefficient shrinkage properties. In such cases, a more computationally efficient solution exists: cyclical coordinate descent (which has a closed-form solution for each descent iteration). For large datasets, or in situations with a large number of predictors, implementation of this solution can lead to substantial speedups. 

A further speedup is implemented for the particular task of pathwise optimization. Since the coefficient path is continuous over <img src="https://latex.codecogs.com/gif.latex?\lambda" />, we implement a "warm start" as we move across the path, pre-initializing the coefficients to the previous solution as different vales of <img src="https://latex.codecogs.com/gif.latex?\lambda" /> are implemented. 

# Versioning
`fast_lasso.py` is compatible with python 3. 

# Usage
Simply import lasso.py and initialize a Lasso object, which takes no arguments. The user then can choose between: 
- Calculating the solutions to a single LASSO model, by calling the `.train()` method on input and output vectors (`numpy` vectors), as well as a single value for <img src="https://latex.codecogs.com/gif.latex?\lambda" />. 
- Creating a graphical solution to the parameter selection problem, by calling the `.lamdbaPath()` method on input and output vectors (`numpy` vectors), as well as a vector of desired values for <img src="https://latex.codecogs.com/gif.latex?\lambda" />.

An example implementation of the latter option is included in `simulation.py`, which compares the output to the stochastic solution included in `sklearn`, which implements the algorithm of [Hastie, Tibshirani and Friedman (2004)](https://projecteuclid.org/euclid.ejs/1177687773). 
