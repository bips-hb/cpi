
#' Conditional Predictive Impact (CPI) for mlr.
#'
#' @param task The prediction task. 
#' @param learner The learner. If you pass a string the learner will be created via \link{makeLearner}.
#' @param resampling Resampling description object, name of resampling strategy, "oob" (out-of-bag) or "none" (in-sample loss).
#' @param test_data External validation data, use instead of resampling.
#' @param measure Performance measure. 
#' @param test Statistical test to perform, either "t" (t-test) or "fisher" (Fisher permuation test).
#' @param permute Permute the feature of interest. Set to \code{FALSE} to drop the feature of interest.
#' @param log Set to \code{TRUE} for multiplicative CPI (\eqn{\lambda}), to \code{FALSE} for additive CPI (\eqn{\Delta}). 
#' @param B Number of permutations for Fisher permutation test.
#' @param alpha Significance level for confidence intervals.
#' @param verbose Verbose output of resampling procedure.
#' @param cores Number CPU cores used.
#'
#' @return \code{data.frame} with a row for each feature and columns:
#'   \item{Variable}{Variable name}
#'   \item{CPI}{CPI value}
#'   \item{SE}{Standard error}
#'   \item{statistic}{Test statistic (t-test only)}
#'   \item{p.value}{p-value}
#'   \item{ci.lo}{Lower limit of confidence interval}
#' 
#' @export
#' @import stats mlr foreach
#'
#' @examples 
#' library(mlr)
#' # Regression with linear model
#' cpi_mlr(task = bh.task, learner = makeLearner("regr.lm"), 
#'         resampling = makeResampleDesc("Holdout"))
#' 
#' # Classification with logistic regression, log-loss and subsampling
#' cpi_mlr(task = iris.task, 
#'         learner = makeLearner("classif.glmnet", predict.type = "prob"), 
#'         resampling = makeResampleDesc("CV", iters = 5), 
#'         measure = "logloss", test = "t")
#'  
#' # Random forest with out-of-bag error               
#' cpi_mlr(task = bh.task, learner = makeLearner("regr.ranger", num.trees = 50), 
#'         resampling = "oob", measure = "mse", test = "t")
#'         
cpi_mlr <- function(task, learner, 
                    resampling = NULL,
                    test_data = NULL,
                    measure = NULL,
                    test = "t",
                    permute = TRUE,
                    log = TRUE,
                    B = 10000,
                    alpha = 0.05, 
                    verbose = FALSE, 
                    cores = 1) {
  if (is.null(measure)) {
    if (getTaskType(task) == "regr") {
      measure <- mse
    } else if (getTaskType(task) == "classif") {
      measure <- logloss
    } else {
      stop("Unknown task type.")
    }
  }
  
  if (is.character(measure)) {
    measure <- eval(parse(text = measure))
  }
  
  if (!is.null(test)) {
    if (!(measure$id %in% c("mse", "mae", "mmce", "logloss", "brier"))) {
      stop("Statistical testing currently only implemented for 'mse', 'mae', 'mmce', 'logloss' and 'brier' measures.")
    }
  }
  
  if (getTaskType(task) == "classif") {
    if (!hasLearnerProperties(learner, "prob")) {
      stop("For classification the learner requires probability support.")
    }
  }
  
  # Create resampling instance
  if (is.null(resampling)) {
    if (is.null(test_data)) {
      stop("Either resampling or test_data argument required.")
    }
  } else if (is.list(resampling)) {
    resample_instance <- makeResampleInstance(desc = resampling, task = task)
  } else if (resampling %in% c("oob", "none")) {
    resample_instance <- resampling
  } else {
    stop("Unknown resampling value.")
  }
  
  # Fit learner and compute performance
  pred_full <- fit_learner(learner = learner, task = task, resampling = resample_instance, measure = measure, test_data = test_data, verbose = verbose)
  aggr_full <- performance(pred_full, measure)
  if (!is.null(test)) {
    err_full <- compute_loss(pred_full, measure)
  }
  
  # For each feature, fit reduced model and return difference in error
  cpi_fun <- function(i) {
    if (permute) {
      reduced_data <- getTaskData(task)
      reduced_data[, getTaskFeatureNames(task)[i]] <- sample(reduced_data[, getTaskFeatureNames(task)[i]])
      reduced_task <- changeData(task, reduced_data)
    } else {
      reduced_task <- subsetTask(task, features = getTaskFeatureNames(task)[-i])
    }
    
    pred_reduced <- fit_learner(learner = learner, task = reduced_task, resampling = resample_instance, measure = measure, test_data = test_data, verbose = verbose)
    aggr_reduced <- performance(pred_reduced, measure)
    
    if (log) { 
      cpi <- log(aggr_reduced / aggr_full)
    } else {
      cpi <- aggr_reduced - aggr_full
    }
    
    res <- data.frame(Variable = getTaskFeatureNames(task)[i],
                      CPI = unname(cpi), 
                      stringsAsFactors = FALSE)

    # Statistical testing
    if (!is.null(test)) {
      err_reduced <- compute_loss(pred_reduced, measure)
      if (log) {
        dif <- log(err_reduced / err_full)
      } else {
        dif <- err_reduced - err_full
      }
      res$CPI <- mean(dif)
      if (test == "fisher") {
        orig_mean <- mean(dif)
        
        # B permutations
        perm_means <- replicate(B, {
          signs <- sample(c(-1, 1), length(dif), replace = TRUE)
          mean(signs * dif)
        })
        res$p.value <- sum(perm_means >= orig_mean)/B
        res$ci.lo <- orig_mean - quantile(perm_means, 1 - alpha)
      } else if (test == "t") {
        test_result <- t.test(dif, alternative = 'greater')
        res$SE <- sd(dif) / sqrt(length(dif))
        res$statistic <- test_result$statistic
        res$p.value <- test_result$p.value
        res$ci.lo <- test_result$conf.int[1]
      } else {
        stop("Unknown test.")
      }
    }
    res
  }
  
  # Run in parallel if >1 cores
  j <- NULL
  if (cores == 1) {
    foreach(j = seq_len(getTaskNFeats(task)), .combine = rbind) %do% cpi_fun(j)
  } else {
    foreach(j = seq_len(getTaskNFeats(task)), .combine = rbind) %dopar% cpi_fun(j)
  }
}

fit_learner <- function(learner, task, resampling = NULL, measure = NULL, test_data = NULL, verbose = FALSE) {
  if (!is.null(test_data)) {
    # Compute error on test data
    mod <- train(learner, task)
    pred <- predict(mod, newdata = test_data)
  } else if (is.list(resampling)) {
    # Full model resampling
    pred <- resample(learner, task, resampling = resampling, measures = measure, show.info = verbose)$pred
  } else if (resampling == "none") {
    # Compute error on training data
    mod <- train(learner, task)
    pred <- predict(mod, task)
  } else if (resampling == "oob") {
    # Use OOB predictions if available
    if (!hasLearnerProperties(learner, "oobpreds")) {
      stop("OOB error not available for this learner.")
    }
    mod <- train(learner, task)
    pred <- getOOBPreds(mod, task)
  } else {
    stop("Unknown value for 'resampling'.")
  }
  pred
}

compute_loss <- function(pred, measure) {
  if (getTaskType(pred) == "regr") {
    if (measure$id == "mse") {
      # Squared errors
      loss <- (pred$data$truth - pred$data$response)^2
    } else if (measure$id == "mae") {
      # Absolute errors
      loss <- abs(pred$data$truth - pred$data$response)
    } else {
      stop("Unknown measure.")
    }
  } else if (getTaskType(pred) == "classif") {
    if (measure$id == "logloss") {
      # Logloss 
      probabilities <- pred$data[, paste("prob", pred$task.desc$class.levels, sep = ".")]
      truth <- match(as.character(pred$data$truth), pred$task.desc$class.levels)
      p <- probabilities[cbind(seq_len(nrow(probabilities)), truth)]
      loss <- -log(p)
    } else if (measure$id == "mmce") {
      # Misclassification error
      loss <- 1*(pred$data$truth != pred$data$response)
    } else if (measure$id == "brier") {
      # Brier score
      y <- as.numeric(pred$data$truth == pred$task.desc$positive)
      loss <- (y - pred$data[, paste("prob", pred$task.desc$positive, sep = ".")])^2
    } else {
      stop("Unknown measure.")
    }
    
    # Avoid 0 and 1
    eps <- 1e-15
    loss[loss > 1 - eps] <- 1 - eps
    loss[loss < eps] <- eps
  } else {
    stop("Unknown task type.")
  }
  
  loss
}
