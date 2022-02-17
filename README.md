<!-- badges: start -->
[![R-CMD-check](https://github.com/dswatson/cpi/workflows/R-CMD-check/badge.svg)](https://github.com/dswatson/cpi/actions)
<!-- badges: end -->

## Conditional Predictive Impact
David S. Watson, Marvin N. Wright

### Introduction
The conditional predictive impact (CPI) is a measure of conditional independence. It can be calculated using any supervised learning algorithm, loss function, and knockoff sampler. We provide statistical inference procedures for the CPI without parametric assumptions or sparsity constraints. The method works with continuous and categorical data.

### Installation
The package is not on CRAN yet. To install the development version from GitHub using `devtools`, run

```R
devtools::install_github("dswatson/cpi")
```

### Examples
Calculate CPI for random forest on iris data with 5-fold cross validation:
```R
library(mlr)
library(cpi)

mytask <- makeClassifTask(data = iris, target = "Species")
cpi(task = mytask, 
    learner = makeLearner("classif.ranger", predict.type = "prob"),
    resampling = makeResampleDesc("CV", iters = 5), 
    measure = "logloss", test = "t")
```

### References
* Watson D. S. & Wright, M. N. (2021). Testing conditional independence in supervised learning algorithms. <em>Machine Learning</em>. DOI: 10.1007/s10994-021-06030-6. 
