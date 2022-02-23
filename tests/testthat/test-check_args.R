library(mlr3)
library(mlr3learners)

test_that("returns object of correct dimensions, regression", {
  task <- tsk("mtcars")
  res <- cpi(task = task, learner = lrn("regr.lm"), 
             resampling = rsmp("holdout"))
  expect_s3_class(res, "data.frame")
  expect_equal(dim(res), c(length(task$feature_names), 8))
  expect_equal(colnames(res), 
               c("Variable", "CPI", "SE", "test", "statistic", "p.value", 
                 "estimate", "ci.lo"))
  expect_equal(res$Variable, 
               task$feature_names)
})

test_that("returns object of correct dimensions, classification", {
  task <- tsk("iris")
  res <- cpi(task = task, 
             learner = lrn("classif.glmnet", predict_type = "prob", lambda = 0.1), 
             resampling = rsmp("cv", folds = 3))
  expect_s3_class(res, "data.frame")
  expect_equal(dim(res), c(length(task$feature_names), 8))
  expect_equal(colnames(res), 
               c("Variable", "CPI", "SE", "test", "statistic", "p.value", 
                 "estimate", "ci.lo"))
  expect_equal(res$Variable, 
               task$feature_names)
})

test_that("returns object of correct dimensions, group classification", {
  groups <- list(Sepal = 1:2, Petal = 3:4)
  res <- cpi(task = tsk("iris"), 
             learner = lrn("classif.ranger", predict_type = "prob"), 
             resampling = rsmp("cv", folds = 3), 
             groups = groups)
  expect_s3_class(res, "data.frame")
  expect_equal(dim(res), c(length(groups), 8))
  expect_equal(colnames(res), 
               c("Group", "CPI", "SE", "test", "statistic", "p.value", 
                 "estimate", "ci.lo"))
  expect_equal(res$Group, 
               names(groups))
})

test_that("fails for wrong groups", {
  expect_error(cpi(task = tsk("iris"), 
      learner = lrn("classif.ranger", predict_type = "prob"), 
      resampling = rsmp("cv", folds = 3), 
      groups = list(a = 1:2, b = 5:6)), 
      "Feature numbers in argument 'groups' not in 1:p, where p is the number of features.")
})

test_that("fails with Gaussian knockoffs and factors", {
  expect_error(cpi(task = tsk("boston_housing"), 
                   learner = lrn("regr.lm"), 
                   resampling = rsmp("holdout")), 
               "Gaussian knockoffs cannot handle factor features\\. Consider using sequential knockoffs \\(see examples\\) or recoding factors\\.")
})
