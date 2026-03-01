# Generative AI Practice

## Goal
My goal in this folder is to learn and implement core **generative modeling concepts** from scratch, understand the math behind them.

## What this folder contains
- `inversetransformsampling.ipynb`  
  Implements inverse transform sampling for the exponential distribution and compares sampled statistics with theoretical values.

- `affine_flow.ipynb`  
  Builds a simple 2D affine normalizing flow (`x = Au + b`) and trains it with maximum likelihood on synthetic data.

- `mnist_affine_flow.ipynb`  
  Extends affine-flow style modeling ideas to MNIST data using PyTorch and includes checkpoint-based training continuation.

- `mvg_em.ipynb`  
  Applies the EM algorithm for multivariate Gaussian data with missing values (imputation-focused setup).

## Learning focus
- Probability distributions and sampling methods
- Maximum likelihood estimation
- Normalizing flow basics
- EM for latent/missing-variable problems
- Practical PyTorch experimentation workflow
