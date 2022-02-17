library(mlr)

test_that("returns object of correct dimensions, regression", {
  bh.task.num <- dropFeatures(bh.task, "chas")
  res <- cpi(task = bh.task.num, learner = makeLearner("regr.lm"), 
             resampling = makeResampleDesc("Holdout"))
  expect_s3_class(res, "data.frame")
  expect_equal(dim(res), c(getTaskNFeats(bh.task.num), 8))
  expect_equal(colnames(res), 
               c("Variable", "CPI", "SE", "test", "statistic", "p.value", 
                 "estimate", "ci.lo"))
  expect_equal(res$Variable, 
               getTaskFeatureNames(bh.task.num))
})

test_that("returns object of correct dimensions, classification", {
  res <- cpi(task = iris.task, 
             learner = makeLearner("classif.glmnet", predict.type = "prob"), 
             resampling = makeResampleDesc("CV", iters = 3))
  expect_s3_class(res, "data.frame")
  expect_equal(dim(res), c(getTaskNFeats(iris.task), 8))
  expect_equal(colnames(res), 
               c("Variable", "CPI", "SE", "test", "statistic", "p.value", 
                 "estimate", "ci.lo"))
  expect_equal(res$Variable, 
               getTaskFeatureNames(iris.task))
})

test_that("returns object of correct dimensions, group classification", {
  groups <- list(Sepal = 1:2, Petal = 3:4)
  res <- cpi(task = iris.task, 
             learner = makeLearner("classif.ranger", predict.type = "prob"), 
             resampling = makeResampleDesc("CV", iters = 3), 
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
  expect_error(cpi(task = iris.task, 
      learner = makeLearner("classif.ranger", predict.type = "prob"), 
      resampling = makeResampleDesc("CV", iters = 3), 
      groups = list(a = 1:2, b = 5:6)), 
      "Feature numbers in argument 'groups' not in 1:p, where p is the number of features.")
})
