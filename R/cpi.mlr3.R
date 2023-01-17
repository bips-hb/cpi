library(mlr3)
library(mlr3proba)
library(mlr3learners)
library(survival)
library(knockoff)
library(devtools)
library(mlr3extralearners)

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
#'    learner will be created via \code{mlr3::\link{lrn}}.
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
#' @param knockoff_fun Function to generate knockoffs. Default: 
#'   \code{knockoff::\link{create.second_order}} with matrix argument.
#' @param groups (Named) list with groups. Set to \code{NULL} (default) for no
#'   groups, i.e. compute CPI for each feature. See examples. 
#' @param verbose Verbose output of resampling procedure.
#' @param t_max Singular value, if specified CPI for survival data is 
#' calculated for \code{seq(t)}
#' @param times Vector of times, to be specified if CPI is calculated for survival data
#' if both times and t_max are not specified, default times are set by \code{seq(min, max, length.out = 100)}
#' @param adjust Set to \code{TRUE} for correcting for multiple testing, 
#' \code{FALSE} (default) for no correction
#' @param method If adjust set to \code{TRUE} method of adjustment is specified, 
#' based on \code{stats::\link{p.adjust}} adjustment methods
#'
#' @return 
#' For \code{test = "bayes"} a list of \code{BEST} objects. In any other 
#' case, a \code{data.frame} with a row for each feature and columns:
#'   \item{Variable/Group}{Variable/group name}
#'   \item{CPI}{CPI value}
#'   \item{SE}{Standard error}
#'   \item{test}{Testing method}
#'   \item{Times}{Times at which CPI is calculated (only in survival data case)}
#'   \item{statistic}{Test statistic (only for t-test, Wilcoxon and binomial test)}
#'   \item{estimate}{Estimated mean (for t-test), median (for Wilcoxon test),
#'     or proportion of \eqn{\Delta}-values greater than 0 (for binomial test).}
#'   \item{p.value}{p-value}
#'   \item{ci.lo}{Lower limit of (1 - \code{alpha}) * 100\% confidence interval}
#' Note that NA values are no error but a result of a CPI value of 0, i.e. no 
#' difference in model performance after replacing a feature with its knockoff.
#'  \item{ci.up}{Upper limit of (1 + \code{alpha}) * 100\% confidence interval}
#' Note that NA values are no error but a result of a CPI value of 0, i.e. no 
#' difference in model performance after replacing a feature with its knockoff.
#' Only in survival data case. 
#'   \item{p.adjust}{adjusted p-value after correction for multiple testing}
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

cpi <- function(task, learner, 
                resampling = NULL,
                test_data = NULL,
                measure = NULL,
                test = "t",
                log = FALSE,
                B = 1999,
                alpha = 0.05, 
                x_tilde = NULL,
                knockoff_fun = function(x) knockoff::create.second_order(as.matrix(x)),
                groups = NULL,
                verbose = FALSE,
                t_max = NULL, 
                times = NULL,
                adjust = FALSE,
                method = NULL) {
  
  # Set verbose level (and save old state)
  old_logger_treshold <- lgr::get_logger("mlr3")$threshold # produces Log Event, which contains a log message along with metadata
  if (verbose) {
    lgr::get_logger("mlr3")$set_threshold("info") 
  } else {
    lgr::get_logger("mlr3")$set_threshold("warn")
  }
  
  if (is.null(measure)) { # check what measure to be used to calculate cpi score based on task type 
    if (task$task_type == "regr") {
      measure <- msr("regr.mse")
    } else if (task$task_type == "classif") {
      measure <- msr("classif.logloss")
    } else if (task$task_type == "surv") {
      measure <- msr("surv.graf")
    } else {
      stop("Unknown task type.")
    }
  }
  if (is.character(measure)) { # if measure is passed by user then use it
    measure <- msr(measure)
  }
  
  if (!(measure$id %in% c("regr.mse", "regr.mae", "classif.ce", "classif.logloss", "classif.bbrier", "surv.graf"))) {
    stop("Currently only implemented for 'regr.mse', 'regr.mae', 'classif.ce', 'classif.logloss', 'classif.bbrier' and 'surv.graf' measures.")
  }
  if (!(test %in% c("t", "fisher", "bayes", "wilcox", "binom"))) {
    stop("Unknown test in 'test' argument.")
  }
  if (test == "bayes") {
    if (!requireNamespace("BEST", quietly = TRUE)) {
      stop("Package \"BEST\" needed for Bayesian testing. Please install it.",
           call. = FALSE)
    }
  }
  
  if (task$task_type == "classif" & measure$id %in% c("classif.logloss", "classif.bbrier")) {
    if (learner$predict_type != "prob") {
      stop("The selected loss function requires probability support. Try predict_type = 'prob' when creating the learner.")
    }
  }
  
  # Check group argument
  if (!is.null(groups)) { # if cpi should be calculated for groups instead if each feature separately
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
  fit_full <- fit_learner(learner = learner,
                          task = task,
                          resampling = resampling,
                          measure = measure,
                          test_data = test_data,
                          verbose = verbose)
  pred_full <- predict_learner(fit_full,
                               task,
                               resampling = resampling,
                               test_data = test_data)
  err_full <- compute_loss(pred_full,
                           measure,
                           task,
                           resampling = resampling,
                           test_data = test_data,
                           t_max = t_max, 
                           times = times)
  
  # Generate knockoff data
  if (is.null(x_tilde)) {
    if (is.null(test_data)) {
      x_tilde <- knockoff_fun(task$data(cols = task$feature_names))
    } else {
      test_data_x_tilde <- knockoff_fun(test_data[, task$feature_names])
    }
  } else if (is.matrix(x_tilde) | is.data.frame(x_tilde)) {
    if (is.null(test_data)) {
      if (any(dim(x_tilde) != dim(task$data(cols = task$feature_names)))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
    } else {
      if (any(dim(x_tilde) != dim(test_data[, task$feature_names]))) {
        stop("Size of 'x_tilde' must match dimensions of data.")
      }
      test_data_x_tilde <- x_tilde
    }
  } else {
    stop("Argument 'x_tilde' must be a matrix, data.frame or NULL.")
  }
  
  # For each feature, fit reduced model and return difference in error
  cpi_fun <- function(i) {
    if (is.null(test_data)) {
      reduced_test_data <- NULL
      reduced_data <- as.data.frame(task$data())
      reduced_data[, task$feature_names[i]] <- x_tilde[, task$feature_names[i]]
      if (task$task_type == "regr") {
        reduced_task <- as_task_regr(reduced_data, target = task$target_names)
      } else if (task$task_type == "classif") {
        reduced_task <- as_task_classif(reduced_data, target = task$target_names)
      } 
    } else {
      reduced_task <- NULL
      reduced_test_data <- test_data
      reduced_test_data[, task$feature_names[i]] <- test_data_x_tilde[, task$feature_names[i]]
    }
    # Predict with knockoff data
    pred_reduced <- predict_learner(fit_full, 
                                    reduced_task, 
                                    resampling = resampling, 
                                    test_data = reduced_test_data)
    err_reduced <- compute_loss(pred_reduced, 
                                measure, 
                                task, 
                                resampling = resampling, 
                                test_data = reduced_test_data,
                                t_max = t_max, 
                                times = times)
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
      res <- list(BEST::BESTmcmc(dif, parallel = FALSE, verbose = FALSE))
      names(res) <- task$feature_names[i]
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
  cpi_fun_surv <- function(i) {
    if (is.null(test_data)) {
      reduced_test_data <- NULL
      reduced_data <- as.data.frame(task$data())
      reduced_data[, task$feature_names[i]] <- x_tilde[, task$feature_names[i]]
      reduced_task <- reduced_task <- as_task_surv(reduced_data, 
                                                   time = task$target_names[1], 
                                                   event = task$target_names[2]) # is this a general way to define time and event for survival data?
    } else {
      reduced_task <- NULL
      reduced_test_data <- test_data
      reduced_test_data[, task$feature_names[i]] <- test_data_x_tilde[, task$feature_names[i]]
    }
    # Predict with knockoff data
    pred_reduced <- predict_learner(fit_full, reduced_task, resampling = resampling, test_data = reduced_test_data)
    err_reduced <- compute_loss(pred_reduced, measure, task, resampling = resampling, test_data = reduced_test_data, times = times, t_max = t_max)
    if (log) {
      dif <- log(err_reduced / err_full)
    } else {
      dif <- err_reduced - err_full
    }
    cpi <- apply(dif, 2, mean, na.rm = TRUE)
    se <- apply(dif, 2, sd, na.rm = TRUE) / apply(dif, 2, function(x) sqrt(length(na.omit(x))))
    
    if (is.null(groups) & (test != "bayes")) {
      res <- data.frame(Variable = rep(task$feature_names[i], length(cpi)),
                        Times = as.numeric(names(cpi)),
                        CPI = unname(cpi), 
                        SE = unname(se),
                        test = rep(unname(test), length(cpi)),
                        stringsAsFactors = FALSE)
    } else if (test != "bayes") {
      res <- data.frame(Group = rep(paste(i, collapse = ","), length(cpi)),
                        CPI = unname(cpi), 
                        SE = unname(se),
                        test = rep(unname(test), length(cpi)),
                        stringsAsFactors = FALSE)
    }
    if (test %in% c('t', 'wilcox', 'binom')){
      statistic = NULL
      estimate = NULL
      p.value = NULL
      ci.lo = NULL
      ci.up = NULL
      for (k in 1:length(cpi)) {
        if (cpi[k] != 0 & !is.na(cpi[k]) & !is.infinite(cpi[k])) {
          if (test == "t") {
            test_result <- t.test(na.omit(dif[,k]), alternative = 'two.sided', 
                                  conf.level = 1 - alpha)
          } else if (test == "wilcox") {
            test_result <- wilcox.test(na.omit(dif[,k]), alternative = 'two.sided', 
                                       conf.int = TRUE,
                                       conf.level = 1 - alpha)
          } else if (test == "binom") {
            test_result <- binom.test(sum(na.omit(dif[,k]) > 0), 
                                      length(na.omit(dif[,k])), 
                                      alternative = 'two.sided', 
                                      conf.level = 1 - alpha)
          } 
          statistic[k] <- test_result$statistic
          estimate[k] <- test_result$estimate
          p.value[k] <- test_result$p.value
          ci.lo[k] <- test_result$conf.int[1] 
          ci.up[k] <- test_result$conf.int[2] 
        }
        else if (cpi[k] == 0 & !is.na(cpi[k]) & !is.infinite(cpi[k])) {
          statistic[k] <- 0
          estimate[k] <- 0
          p.value[k] <- 1
          ci.lo[k] <- 0 
          ci.up[k] <- 0 
        }
        else {
          statistic[k] <- NA
          estimate[k] <- NA
          p.value[k] <- NA
          ci.lo[k] <- NA  
          ci.up[k] <- NA
        }
      }
      res$statistic <- statistic
      res$estimate <- estimate
      res$p.value <- p.value
      res$ci.lo <- ci.lo
      res$ci.up <- ci.up
    } else if (test == "fisher") {
      p.value <- NA
      ci.lo <- NA
      ci.up <- NA
      for (k in 1:length(cpi)) {
        if (is.na(cpi[k]) | all(is.na(dif[,k])) | is.infinite(cpi[k])) {
          p.value[k] <- NA
          ci.lo[k] <- NA
          ci.up[k] <- NA
        } else { 
          orig_mean = mean(na.omit(dif[,k]))
          # B permutations
          perm_means <- replicate(B, {
            signs <- sample(c(-1, 1), length(na.omit(dif[,k])), replace = TRUE)
            mean(signs * na.omit(dif[,k]))
          })
          p.value[k] <- (sum(perm_means >= orig_mean) + 1)/(B + 1)
          ci.lo[k] <- orig_mean - quantile(perm_means, 1 - alpha)
          ci.up[k] <- orig_mean + quantile(perm_means, 1 - alpha)
        }}
      res$p.value <- p.value
      res$ci.lo <- ci.lo
      res$ci.up <- ci.up
    } else if (test == "bayes") {
      if (!exists("res")) {
        res = NULL
      }
      res_inner = NULL
      l = 0
      for (k in 1:length(cpi)) {
        if (!is.na(cpi[k]) & cpi[k] != 0 & !is.infinite(cpi[k])) {
          l <- l + 1
          res_indiv <- list(BEST::BESTmcmc(na.omit(dif[,k]), parallel = FALSE, verbose = FALSE))
          res_inner <- append(res_inner, res_indiv)
          names(res_inner)[l] <- colnames(dif)[k]
        }}
      res[[i]] <- res_inner
      names(res)[i] <- task$feature_names[i]
      res <- Filter(function(a) any(!is.null(a)), res)
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
  if (task$task_type %in% c('regr', 'classif')) {
    if (foreach::getDoParRegistered()) {
      ret <- foreach(j = idx, .combine = .combine) %dopar% cpi_fun(j)
    } else {
      ret <- foreach(j = idx, .combine = .combine) %do% cpi_fun(j)
    }
  } else if (task$task_type == "surv") {
    if (foreach::getDoParRegistered()) {
      ret <- foreach(j = idx, .combine = .combine) %dopar% cpi_fun_surv(j)
    } else {
      ret <- foreach(j = idx, .combine = .combine) %do% cpi_fun_surv(j)
    }
  } else {stop("Unknown task type.")}
  
  if (test != "bayes" & adjust == TRUE & method %in% c("holm", "hochberg", 
                                                       "hommel", "bonferroni", 
                                                       "BH", "BY", "fdr")){
    ret$p.adjust = p.adjust(ret$p.value, method = method)
  } else if (!(method %in% c("holm", "hochberg", 
                          "hommel", "bonferroni", 
                          "BH", "BY", "fdr"))) {
    stop("Unknown method to adjust for multiple testing.")
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

compute_loss <- function(pred, measure, task, resampling = NULL, 
                         test_data = NULL, t_max = NULL, times = NULL) {
  if (inherits(pred, "Prediction")) {
    truth <- pred$truth
    response <- pred$response
    prob <- pred$prob
  } else {
    truth <- do.call(c, lapply(pred, function(x)
      x$truth))
    response <- do.call(c, lapply(pred, function(x)
      x$response))
    prob <- do.call(rbind, lapply(pred, function(x)
      x$prob))
  }
  if (measure$id == "regr.mse") {
    # Squared errors
    loss <- (truth - response) ^ 2
  } else if (measure$id == "regr.mae") {
    # Absolute errors
    loss <- abs(truth - response)
  } else if (measure$id == "classif.logloss") {
    # Logloss
    eps <- 1e-15
    ii <- match(as.character(truth), colnames(prob))
    p <- prob[cbind(seq_len(nrow(prob)), ii)]
    p <- pmax(eps, pmin(1 - eps, p))
    loss <- -log(p)
  } else if (measure$id == "classif.ce") {
    # Misclassification error
    loss <- 1 * (truth != response)
  } else if (measure$id == "classif.bbrier") {
    # Brier score
    # First level is positive class
    y <- as.numeric(as.numeric(truth) == 1)
    loss <- (y - prob[, 1]) ^ 2
  } else if (measure$id == "surv.graf") {
    # Survival Brier score
    loss <- survival_loss(loss = "graf",
                          pred = pred, 
                          t_max = t_max, 
                          times = times,
                          resampling = resampling, 
                          test_data = test_data)
  } else if (measure$id == "surv.schmid") {
    # Survival Brier score
    loss <- survival_loss(loss = "schmid",
                          pred = pred, 
                          t_max = t_max,  
                          times = times,
                          resampling = resampling, 
                          test_data = test_data)
  } else if (measure$id == "surv.logloss") {
    # Survival Brier score
    loss <- survival_loss(loss = "logloss",
                          pred = pred, 
                          t_max = t_max,  
                          times = times,
                          resampling = resampling, 
                          test_data = test_data)
  } else {
    stop("Unknown measure.")
  }
  loss
}




