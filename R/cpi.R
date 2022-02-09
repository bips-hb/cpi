#' Conditional Predictive Impact (CPI)
#'
#' @param task The prediction task. 
#' @param learner The learner. If you pass a string the learner will be created 
#'   via \link{makeLearner}.
#' @param resampling Resampling description object, mlr resampling strategy 
#'   (e.g. \code{makeResampleDesc("Holdout")}), "oob" (out-of-bag) or "none" 
#'   (in-sample loss).
#' @param test_data External validation data, use instead of resampling.
#' @param measure Performance measure. 
#' @param test Statistical test to perform, one of \code{"t"} (t-test, default), 
#'   \code{"wilcox"} (Wilcoxon signed-rank test), \code{"binom"} (binomial 
#'   test), \code{"fisher"} (Fisher permutation test) or "bayes" 
#'   (Bayesian testing, computationally intensive!). See Details.
#' @param log Set to \code{TRUE} for multiplicative CPI (\eqn{\lambda}), to 
#'   \code{FALSE} for additive CPI (\eqn{\Delta}). 
#' @param B Number of permutations for Fisher permutation test.
#' @param alpha Significance level for confidence intervals.
#' @param x_tilde Knockoff matrix. If not given (the default), it will be 
#'   created with \link{create.second_order}.
#' @param verbose Verbose output of resampling procedure.
#' @param cores Number of CPU cores used.
#'
#' @return For \code{test = "bayes"} a list of \code{BEST} objects. In any other 
#'   case, a \code{data.frame} with a row for each feature and columns:
#'   \item{Variable}{Variable name}
#'   \item{CPI}{CPI value}
#'   \item{SE}{Standard error}
#'   \item{test}{Testing method}
#'   \item{statistic}{Test statistic (only for t-test)}
#'   \item{p.value}{p-value}
#'   \item{estimate}{Estimated mean (for t-test), median (for Wilcoxon test),
#'     or proportion of \eqn{\Delta}-values greater than 0 (for binomial test).}
#'   \item{ci.lo}{Lower limit of (1 - \code{alpha}) * 100% confidence interval}
#' 
#' @export
#' @import stats mlr foreach
#' @importFrom knockoff create.second_order
#'
#' @details 
#' This function computes the conditional predictive impact (CPI) of one or
#' several features on a given supervised learning task. This represents the 
#' mean error inflation when replacing a true variable with its knockoff. Large
#' CPI values are evidence that the feature(s) in question have high 
#' \emph{conditional variable importance} -- i.e., the fitted model relies on 
#' the feature(s) to predict the outcome, even after accounting for the signal
#' from all remaining covariates. 
#' 
#' We build on the \code{mlr} framework, which provides a unified interface for 
#' training models, specifying loss functions, and estimating generalization 
#' error. See the package documentation for more info.
#' 
#' Methods are implemented for frequentist and Bayesian inference. The default
#' is \code{test = "t"}, which is fast and powerful for most sample sizes. The
#' Wilcoxon signed-rank test may be more appropriate if the CPI distribution is 
#' skewed, while the binomial test requires basically no assumptions but may
#' have less power. For small sample sizes, we recommend permutation tests 
#' (\code{test = "fisher"}) or Bayesian methods (\code{test = "bayes"}). In
#' the latter case, default priors are assumed. See the \code{BEST} package for
#' more info.
#' 
#' @references
#' Watson, D. & Wright, M. (2020). Testing conditional independence in 
#' supervised learning algorithms. \emph{Machine Learning}, \emph{110}(8): 
#' 2107-2129. \href{https://link.springer.com/article/10.1007%2Fs10994-021-06030-6}{URL}
#' 
#' Candès, E., Fan, Y., Janson, L, & Lv, J. (2018). {Panning for gold: 'model-X'
#' knockoffs for high dimensional controlled variable selection}. \emph{J. R. 
#' Statistc. Soc. B}, \emph{80}(3): 551-577. \href{https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12265}{URL}
#'
#' @examples 
#' library(mlr)
#' # Regression with linear model
#' bh.task.num <- dropFeatures(bh.task, "chas")
#' cpi(task = bh.task.num, learner = makeLearner("regr.lm"), 
#'     resampling = makeResampleDesc("Holdout"))
#' 
#' # Classification with logistic regression, log-loss and subsampling
#' cpi(task = iris.task, 
#'     learner = makeLearner("classif.glmnet", predict.type = "prob"), 
#'     resampling = makeResampleDesc("CV", iters = 5), 
#'     measure = "logloss", test = "t")
#'  
#' # Use your own data
#' mytask <- makeClassifTask(data = iris, target = "Species")
#' mylearner <- makeLearner("classif.ranger")
#' cpi(task = mytask, learner = mylearner, 
#'     resampling = makeResampleDesc("Subsample", iters = 5), 
#'     measure = "mmce", test = "fisher")
#'     
#' \dontrun{
#' # Bayesian testing
#' res <- cpi(task = iris.task, 
#'            learner = makeLearner("classif.glmnet", predict.type = "prob"), 
#'            resampling = makeResampleDesc("Holdout"), 
#'            measure = "logloss", test = "bayes")
#' plot(res$Petal.Length)
#' }   
#' 
cpi <- function(task, learner, 
                resampling = NULL,
                test_data = NULL,
                measure = NULL,
                test = "t",
                log = FALSE,
                B = 1999,
                alpha = 0.05, 
                x_tilde = NULL,
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
  
  if (!(measure$id %in% c("mse", "mae", "mmce", "logloss", "brier"))) {
    stop("Currently only implemented for 'mse', 'mae', 'mmce', 'logloss' and 'brier' measures.")
  }
  if (!(test %in% c("t", "fisher"))) {
    stop("Currently only t-test (\"t\") and Fisher's exact test (\"fisher\") implemented.")
  }
  if (test == "bayes") {
    if (!requireNamespace("BEST", quietly = TRUE)) {
      stop("Package \"BEST\" needed for Bayesian testing. Please install it.",
           call. = FALSE)
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
  } else if (resampling %in% c("none", "oob")) {
    resample_instance <- resampling
  } else {
    stop("Unknown resampling value.")
  }
  
  # Fit learner and compute performance
  fit_full <- fit_learner(learner = learner, task = task, resampling = resample_instance, 
                          measure = measure, test_data = test_data, verbose = verbose)
  pred_full <- predict_learner(fit_full, task, resampling = resample_instance, test_data = test_data)
  err_full <- compute_loss(pred_full, measure)
  
  # Generate knockoff data
  if (is.null(x_tilde)) {
    if (is.null(test_data)) {
      x_tilde <- knockoff::create.second_order(as.matrix(getTaskData(task)[, getTaskFeatureNames(task)]))
    } else {
      test_data_x_tilde <- knockoff::create.second_order(as.matrix(test_data[, getTaskFeatureNames(task)]))
    }
  } else if (is.matrix(x_tilde)) {
    if (is.null(test_data)) {
      if (any(dim(x_tilde) != dim(as.matrix(getTaskData(task)[, getTaskFeatureNames(task)])))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
    } else {
      if (any(dim(x_tilde) != dim(as.matrix(test_data[, getTaskFeatureNames(task)])))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
      test_data_x_tilde <- x_tilde
    }
  } else {
    stop("Argument 'x_tilde' must be a matrix or NULL.")
  }

  # For each feature, fit reduced model and return difference in error
  cpi_fun <- function(i) {
    if (is.null(test_data)) {
      reduced_test_data <- NULL
      reduced_data <- getTaskData(task)
      reduced_data[, getTaskFeatureNames(task)[i]] <- x_tilde[, getTaskFeatureNames(task)[i]]
      reduced_task <- changeData(task, reduced_data)
    } else {
      reduced_task <- NULL
      reduced_test_data <- test_data
      reduced_test_data[, getTaskFeatureNames(task)[i]] <- test_data_x_tilde[, getTaskFeatureNames(task)[i]]
    }
    
    # Predict with knockoff data
    pred_reduced <- predict_learner(fit_full, reduced_task, resampling = resample_instance, test_data = reduced_test_data)
    err_reduced <- compute_loss(pred_reduced, measure)
    if (log) {
      dif <- log(err_reduced / err_full)
    } else {
      dif <- err_reduced - err_full
    }
    cpi <- mean(dif)
    se <- sd(dif) / sqrt(length(dif))
    
    res <- data.frame(Variable = getTaskFeatureNames(task)[i],
                      CPI = unname(cpi), 
                      SE = unname(se),
                      test = unname(test),
                      stringsAsFactors = FALSE)
    
    # Statistical testing
    if (test == "fisher") {
      orig_mean <- mean(dif)
      # B permutations
      perm_means <- replicate(B, {
        signs <- sample(c(-1, 1), length(dif), replace = TRUE)
        mean(signs * dif)
      })
      res$p.value <- (sum(perm_means >= orig_mean) + 1)/(B + 1)
      res$ci.lo <- orig_mean - quantile(perm_means, 1 - alpha)
    } else if (test == "bayes") {
      res <- list(BEST::BESTmcmc(dif, parallel = FALSE, verbose = FALSE))
      names(res) <- getTaskFeatureNames(task)[i]
    } else if (test %in% c('t', 'wilcox', 'binom')) {
      if (test == "t") {
      test_result <- t.test(dif, alternative = 'greater', 
                            conf.level = 1 - alpha)
      res$statistic <- test_result$statistic
      } else if (test == "wilcox") {
        test_result <- wilcox.test(dif, alternative = 'greater', conf.int = TRUE,
                                   conf.level = 1 - alpha)
      } else if (test == "binom") {
        test_result <- binom.test(sum(dif > 0), length(dif), alternative = 'greater', 
                                  conf.level = 1 - alpha)
      } 
      res$p.value <- test_result$p.value
      res$estimate <- test_result$estimate
      res$ci.lo <- test_result$conf.int[1]
    } else {
      stop("Unknown test.")
    }
    res
  }
  
  # Different return value for Bayesian testing
  if (test == "bayes") {
    .combine = c
  } else {
    .combine = rbind
  }
  
  # Run in parallel if >1 cores
  j <- NULL
  if (cores == 1) {
    foreach(j = seq_len(getTaskNFeats(task)), .combine = .combine) %do% cpi_fun(j)
  } else {
    foreach(j = seq_len(getTaskNFeats(task)), .combine = .combine) %dopar% cpi_fun(j)
  }
}




