
## Conditional Predictive Impact
David S. Watson, Marvin N. Wright

### Introduction
Conditional Predictive Impact (CPI) is a general test for conditional independence in supervised learning algorithms. The measure can be calculated using any supervised learning algorithm and loss function. It provides statistical inference procedures without parametric assumptions and applies equally well to continuous and categorical predictors and outcomes.

### Installation
The package is not on CRAN yet. To install the development version from GitHub using `devtools`, run

```R
devtools::install_github("dswatson/cpi")
```

### Examples
Calculate CPI for random forest on iris data with 5-fold cross validation:
```R
mytask <- makeClassifTask(data = iris, target = "Species")
cpi(task = mytask, 
    learner = makeLearner("classif.ranger", num.trees = 50),
    resampling = makeResampleDesc("CV", iters = 5), 
    measure = "mmce", test = "t")
```

### References
* Watson D. S. & Wright, M. N. (2018). Testing conditional independence in supervised learning algorithms. In preparation. 
