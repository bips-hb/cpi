#' Conditional Predictive Impact (CPI). 
#' 
#' A general test for conditional 
#' independence in supervised learning algorithms. Implements a conditional 
#' variable importance measure which can be applied to any supervised learning 
#' algorithm and loss function. Provides statistical inference procedures 
#' without parametric assumptions and applies equally well to continuous and 
#' categorical predictors and outcomes.
#'
#' @param task The prediction \code{mlr3} task, see examples.
#' @param learner The \code{mlr3} learner used in CPI. If you pass a string, the 
#'    learner will be created via \code{mlr3::\link[mlr3]{lrn}}.
#' @param resampling Resampling strategy, \code{mlr3} resampling object 
#'   (e.g. \code{rsmp("holdout")}), "oob" (out-of-bag) or "none" 
#'   (in-sample loss).
#' @param test_data External validation data, use instead of resampling.
#' @param measure Performance measure (loss). Per default, use MSE 
#'    (\code{"regr.mse"}) for regression and logloss (\code{"classif.logloss"}) 
#'    for classification. 
#' @param test Statistical test to perform, one of \code{"t"} (t-test, default), 
#'   \code{"wilcox"} (Wilcoxon signed-rank test), \code{"binom"} (binomial 
#'   test), \code{"fisher"} (Fisher permutation test) or "bayes" 
#'   (Bayesian testing, computationally intensive!). See Details.
#' @param log Set to \code{TRUE} for multiplicative CPI (\eqn{\lambda}), to 
#'   \code{FALSE} (default) for additive CPI (\eqn{\Delta}). 
#' @param B Number of permutations for Fisher permutation test.
#' @param alpha Significance level for confidence intervals.
#' @param x_tilde Knockoff matrix or data.frame. If not given (the default), it will be 
#'   created with the function given in \code{knockoff_fun}. 
#'   Also accepts a list of matrices or data.frames.
#' @param aggr_fun Aggregation function over replicates. 
#' @param knockoff_fun Function to generate knockoffs. Default: 
#'   \code{knockoff::\link[knockoff]{create.second_order}} with matrix argument.
#' @param groups (Named) list with groups. Set to \code{NULL} (default) for no
#'   groups, i.e. compute CPI for each feature. See examples. 
#' @param verbose Verbose output of resampling procedure.
#'
#' @return 
#' For \code{test = "bayes"} a list of \code{BEST} objects. In any other 
#' case, a \code{data.frame} with a row for each feature and columns:
#'   \item{Variable/Group}{Variable/group name}
#'   \item{CPI}{CPI value}
#'   \item{SE}{Standard error}
#'   \item{test}{Testing method}
#'   \item{statistic}{Test statistic (only for t-test, Wilcoxon and binomial test)}
#'   \item{estimate}{Estimated mean (for t-test), median (for Wilcoxon test),
#'     or proportion of \eqn{\Delta}-values greater than 0 (for binomial test).}
#'   \item{p.value}{p-value}
#'   \item{ci.lo}{Lower limit of (1 - \code{alpha}) * 100\% confidence interval}
#' Note that NA values are no error but a result of a CPI value of 0, i.e. no 
#' difference in model performance after replacing a feature with its knockoff.
#' @export
#' @import stats mlr3 foreach
#' @importFrom knockoff create.second_order
#' @importFrom lgr get_logger
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
#' We build on the \code{mlr3} framework, which provides a unified interface for 
#' training models, specifying loss functions, and estimating generalization 
#' error. See the package documentation for more info.
#' 
#' Methods are implemented for frequentist and Bayesian inference. The default
#' is \code{test = "t"}, which is fast and powerful for most sample sizes. The
#' Wilcoxon signed-rank test (\code{test = "wilcox"}) may be more appropriate if 
#' the CPI distribution is skewed, while the binomial test (\code{test = "binom"}) 
#' requires basically no assumptions but may have less power. For small sample 
#' sizes, we recommend permutation tests (\code{test = "fisher"}) or Bayesian 
#' methods (\code{test = "bayes"}). In the latter case, default priors are 
#' assumed. See the \code{BEST} package for more info.
#' 
#' For parallel execution, register a backend, e.g. with
#' \code{doParallel::registerDoParallel()}.
#' 
#' @references
#' Watson, D. & Wright, M. (2020). Testing conditional independence in 
#' supervised learning algorithms. \emph{Machine Learning}, \emph{110}(8): 
#' 2107-2129. \doi{10.1007/s10994-021-06030-6}
#' 
#' Candès, E., Fan, Y., Janson, L, & Lv, J. (2018). {Panning for gold: 'model-X'
#' knockoffs for high dimensional controlled variable selection}. \emph{J. R. 
#' Statistc. Soc. B}, \emph{80}(3): 551-577. \doi{10.1111/rssb.12265}
#'
#' @examples 
#' library(mlr3)
#' library(mlr3learners)
#' 
#' # Regression with linear model and holdout validation
#' cpi(task = tsk("mtcars"), learner = lrn("regr.lm"), 
#'     resampling = rsmp("holdout"))
#' 
#' \donttest{
#' # Classification with logistic regression, log-loss and t-test
#' cpi(task = tsk("wine"), 
#'     learner = lrn("classif.glmnet", predict_type = "prob", lambda = 0.1), 
#'     resampling = rsmp("holdout"), 
#'     measure = "classif.logloss", test = "t")
#'  
#' # Use your own data (and out-of-bag loss with random forest)
#' mytask <- as_task_classif(iris, target = "Species")
#' mylearner <- lrn("classif.ranger", predict_type = "prob", keep.inbag = TRUE)
#' cpi(task = mytask, learner = mylearner, 
#'     resampling = "oob", measure = "classif.logloss")
#'     
#' # Group CPI
#' cpi(task = tsk("iris"), 
#'     learner = lrn("classif.ranger", predict_type = "prob", num.trees = 10), 
#'     resampling = rsmp("cv", folds = 3), 
#'     groups = list(Sepal = 1:2, Petal = 3:4))
#' }     
#' \dontrun{      
#' # Bayesian testing
#' res <- cpi(task = tsk("iris"), 
#'            learner = lrn("classif.glmnet", predict_type = "prob", lambda = 0.1), 
#'            resampling = rsmp("holdout"), 
#'            measure = "classif.logloss", test = "bayes")
#' plot(res$Petal.Length)
#' 
#' # Parallel execution
#' doParallel::registerDoParallel()
#' cpi(task = tsk("wine"), 
#'     learner = lrn("classif.glmnet", predict_type = "prob", lambda = 0.1), 
#'     resampling = rsmp("cv", folds = 5))
#'     
#' # Use sequential knockoffs for categorical features
#' # package available here: https://github.com/kormama1/seqknockoff
#' mytask <- as_task_regr(iris, target = "Petal.Length")
#' cpi(task = mytask, learner = lrn("regr.ranger"), 
#'     resampling = rsmp("holdout"), 
#'     knockoff_fun = seqknockoff::knockoffs_seq)
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
                aggr_fun = mean,
                knockoff_fun = function(x) knockoff::create.second_order(as.matrix(x)),
                groups = NULL,
                verbose = FALSE) {
  
  # Set verbose level (and save old state)
  old_logger_treshold <- lgr::get_logger("mlr3")$threshold
  if (verbose) {
    lgr::get_logger("mlr3")$set_threshold("info")
  } else {
    lgr::get_logger("mlr3")$set_threshold("warn")
  }
  
  if (is.null(measure)) {
    if (task$task_type == "regr") {
      measure <- msr("regr.mse")
    } else if (task$task_type == "classif") {
      measure <- msr("classif.logloss")
    } else {
      stop("Unknown task type.")
    }
  }
  if (is.character(measure)) {
    measure <- msr(measure)
  }
  
  if (!(measure$id %in% c("regr.mse", "regr.mae", "classif.ce", "classif.logloss", "classif.bbrier"))) {
    stop("Currently only implemented for 'regr.mse', 'regr.mae', 'classif.ce', 'classif.logloss' and 'classif.bbrier' measures.")
  }
  if (!(test %in% c("t", "fisher", "bayes", "wilcox", "binom"))) {
    stop("Unknown test in 'test' argument.")
  }
  if (test == "bayes") {
    # if (!requireNamespace("BEST", quietly = TRUE)) {
    #   stop("Package \"BEST\" needed for Bayesian testing. Please install it.",
    #        call. = FALSE)
    # }
    stop("Bayesian testing currently not implemented as BEST package was removed from CRAN.",
         call. = FALSE)
  }
  
  if (task$task_type == "classif" & measure$id %in% c("classif.logloss", "classif.bbrier")) {
    if (learner$predict_type != "prob") {
      stop("The selected loss function requires probability support. Try predict_type = 'prob' when creating the learner.")
    }
  }
  
  # Check group argument
  if (!is.null(groups)) {
    if (!is.list(groups)) {
      stop("Argument 'groups' is expected to be a (named) list with feature numbers, see examples.")
    }
    if (max(unlist(groups)) > length(task$feature_names) | any(unlist(groups) < 1)) {
      stop("Feature numbers in argument 'groups' not in 1:p, where p is the number of features.")
    }
  }
  
  # Check knockoffs
  if (any(task$feature_types$type == "factor") && is.null(x_tilde) && is.function(knockoff_fun) && deparse(knockoff_fun)[2] == "knockoff::create.second_order(as.matrix(x))") {
    stop("Gaussian knockoffs cannot handle factor features. Consider using sequential knockoffs (see examples) or recoding factors.")
  }
  
  # Create resampling instance
  if (is.null(resampling)) {
    if (is.null(test_data)) {
      stop("Either resampling or test_data argument required.")
    }
  } else if (inherits(resampling, "Resampling")) {
    resampling$instantiate(task)
  } else if (resampling %in% c("none", "oob")) {
    # Do nothing
  } else {
    stop("Unknown resampling value.")
  }
  
  # Fit learner and compute performance
  fit_full <- fit_learner(learner = learner, task = task, resampling = resampling, 
                          measure = measure, test_data = test_data, verbose = verbose)
  pred_full <- predict_learner(fit_full, task, resampling = resampling, test_data = test_data)
  err_full <- compute_loss(pred_full, measure)
  
  # Generate knockoff data
  if (is.null(x_tilde)) {
    if (is.null(test_data)) {
      x_tilde <- knockoff_fun(task$data(cols = task$feature_names))
    } else {
      x_tilde <- knockoff_fun(test_data[, task$feature_names])
    }
    x_tilde <- list(x_tilde)
  } else if (is.matrix(x_tilde) | is.data.frame(x_tilde)) {
    if (is.null(test_data)) {
      if (any(dim(x_tilde) != dim(task$data(cols = task$feature_names)))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
    } else {
      if (any(dim(x_tilde) != dim(test_data[, task$feature_names]))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
    }
    x_tilde <- list(x_tilde)
  } else if (is.list(x_tilde)) {
    if (length(x_tilde) < 1) {
      stop("If 'x_tilde' is a list, it cannot be empty.")
    }
    if (is.null(test_data)) {
      #FIXME: Check all dims
      if (any(dim(x_tilde[[1]]) != dim(task$data(cols = task$feature_names)))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
    } else {
      #FIXME: Check all dims
      if (any(dim(x_tilde[[1]]) != dim(test_data[, task$feature_names]))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
    }
  } else {
    stop("Argument 'x_tilde' must be a matrix, data.frame or NULL.")
  }

  # For each feature, fit reduced model and return difference in error
  cpi_fun <- function(i) {
    err_reduced <- sapply(x_tilde, function(x_tilde_i) {
      if (is.null(test_data)) {
        reduced_test_data <- NULL
        reduced_data <- as.data.frame(task$data())
        reduced_data[, task$feature_names[i]] <- x_tilde_i[, task$feature_names[i]]
        if (task$task_type == "regr") {
          reduced_task <- as_task_regr(reduced_data, target = task$target_names)
        } else if (task$task_type == "classif") {
          reduced_task <- as_task_classif(reduced_data, target = task$target_names)
        } else {
          stop("Unknown task type.")
        }
      } else {
        reduced_task <- NULL
        reduced_test_data <- test_data
        reduced_test_data[, task$feature_names[i]] <- x_tilde_i[, task$feature_names[i]]
      }
      
      # Predict with knockoff data
      pred_reduced <- predict_learner(fit_full, reduced_task, resampling = resampling, test_data = reduced_test_data)
      err_reduced <- compute_loss(pred_reduced, measure)
      
      err_reduced
    })
    
    # Average over results with different knockoffs
    err_reduced <- apply(err_reduced, 1, aggr_fun)
    
    if (log) {
      dif <- log(err_reduced / err_full)
    } else {
      dif <- err_reduced - err_full
    }
    cpi <- mean(dif)
    se <- sd(dif) / sqrt(length(dif))

    if (is.null(groups)) {
      res <- data.frame(Variable = task$feature_names[i],
                        CPI = unname(cpi), 
                        SE = unname(se),
                        test = unname(test),
                        stringsAsFactors = FALSE)
    } else {
      res <- data.frame(Group = paste(i, collapse = ","),
                        CPI = unname(cpi), 
                        SE = unname(se),
                        test = unname(test),
                        stringsAsFactors = FALSE)
    }
    
    # Statistical testing
    if (cpi == 0) {
      # No test if CPI==0
      if (test != "bayes") {
        if (test %in% c('t', 'wilcox', 'binom')) {
          res$statistic <- 0
          res$estimate <- 0
        }
        res$p.value <- 1
        res$ci.lo <- 0
      }
    } else if (test == "fisher") {
      orig_mean <- mean(dif)
      # B permutations
      perm_means <- replicate(B, {
        signs <- sample(c(-1, 1), length(dif), replace = TRUE)
        mean(signs * dif)
      })
      res$p.value <- (sum(perm_means >= orig_mean) + 1)/(B + 1)
      res$ci.lo <- orig_mean - quantile(perm_means, 1 - alpha)
    } else if (test == "bayes") {
      #res <- list(BEST::BESTmcmc(dif, parallel = FALSE, verbose = FALSE))
      #names(res) <- task$feature_names[i]
      message("Bayesian testing currently not implemented as BEST package was removed from CRAN.")
    } else if (test %in% c('t', 'wilcox', 'binom')) {
      if (test == "t") {
        test_result <- t.test(dif, alternative = 'greater', 
                              conf.level = 1 - alpha)
      } else if (test == "wilcox") {
        test_result <- wilcox.test(dif, alternative = 'greater', conf.int = TRUE,
                                   conf.level = 1 - alpha)
      } else if (test == "binom") {
        test_result <- binom.test(sum(dif > 0), length(dif), alternative = 'greater', 
                                  conf.level = 1 - alpha)
      }
      res$statistic <- test_result$statistic
      res$estimate <- test_result$estimate
      res$p.value <- test_result$p.value
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
  
  # If group CPI, iterate over groups
  if (is.null(groups)) {
    idx <- seq_len(length(task$feature_names))
  } else {
    idx <- groups
  }
  
  # Run in parallel if a parallel backend is registered
  j <- NULL
  if (foreach::getDoParRegistered()) {
    ret <- foreach(j = idx, .combine = .combine) %dopar% cpi_fun(j)
  } else {
    ret <- foreach(j = idx, .combine = .combine) %do% cpi_fun(j)
  }
  
  # If group CPI, rename groups
  if (!is.null(groups) & !is.null(names(groups))) {
    ret$Group <- names(groups)
  }
  
  # Reset to old logging threshold
  lgr::get_logger("mlr3")$set_threshold(old_logger_treshold)
  
  # Return CPI for all features/groups
  ret
}




