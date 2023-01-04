library(mlr3)
library(mlr3proba)
library(mlr3learners)
library(survival)
library(knockoff)
library(devtools)
library(mlr3extralearners)

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
                     verbose = FALSE) {
  
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
                           test_data = test_data)
  
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
                                test_data = reduced_test_data)
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
    err_reduced <- compute_loss(pred_reduced, measure, task, resampling = resampling, test_data = reduced_test_data)
    if (log) {
      dif <- log(err_reduced / err_full)
    } else {
      dif <- err_reduced - err_full
    }
    cpi <- apply(dif, 2, mean, na.rm = TRUE)
    se <- apply(dif, 2, sd, na.rm = TRUE) / apply(dif, 2, function(x) sqrt(length(na.omit(x))))
    
    if (is.null(groups)) {
      res <- data.frame(Variable = rep(task$feature_names[i], length(cpi)),
                        Times = as.numeric(names(cpi)),
                        CPI = unname(cpi), 
                        SE = unname(se),
                        test = rep(unname(test), length(cpi)),
                        stringsAsFactors = FALSE)
    } else {
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
      for (k in 1:length(cpi)) {
        if (cpi[k] != 0 & !is.na(cpi[k])) {
          if (test == "t") {
            test_result <- t.test(na.omit(dif[,k]), alternative = 'greater', 
                                  conf.level = 1 - alpha)
          } else if (test == "wilcox") {
            test_result <- wilcox.test(na.omit(dif[,k]), alternative = 'greater', 
                                       conf.int = TRUE,
                                       conf.level = 1 - alpha)
          } else if (test == "binom") {
            test_result <- binom.test(sum(na.omit(dif[,k]) > 0), 
                                      length(na.omit(dif[,k])), 
                                      alternative = 'greater', 
                                      conf.level = 1 - alpha)
          } 
          statistic[k] <- test_result$statistic
          estimate[k] <- test_result$estimate
          p.value[k] <- test_result$p.value
          ci.lo[k] <- test_result$conf.int[1] 
        }
        else if (cpi[k] == 0 & !is.na(cpi[k])) {
          statistic[k] <- 0
          estimate[k] <- 0
          p.value[k] <- 1
          ci.lo[k] <- 0 
        }
        else {
          statistic[k] <- NA
          estimate[k] <- NA
          p.value[k] <- NA
          ci.lo[k] <- NA       
        }
      }
      res$statistic <- statistic
      res$estimate <- estimate
      res$p.value <- p.value
      res$ci.lo <- ci.lo
    } else if (test == "fisher") {
      p.value[k] <- NA
      ci.lo[k] <- NA
      for (k in 1:length(cpi)) {
        if (is.na(cpi[k]) | all(is.na(dif[,k]))) {
          p.value[k] <- NA
          ci.lo[k] <- NA
        } else { 
          orig_mean = mean(na.omit(dif[,k]))
          # B permutations
          perm_means <- replicate(B, {
            signs <- sample(c(-1, 1), length(na.omit(dif[,k])), replace = TRUE)
            mean(signs * na.omit(dif[,k]))
          })
          p.value[k] <- (sum(perm_means >= orig_mean) + 1)/(B + 1)
          ci.lo[k] <- orig_mean - quantile(perm_means, 1 - alpha)
        }}
      res$p.value <- p.value
      res$ci.lo <- ci.lo
    } else if (test == "bayes") {
      res_inner = NULL
      for (k in 1:length(cpi)) {
        if (!is.na(cpi[k]) & cpi[k] != 0) {
          res_inner <- list(BEST::BESTmcmc(na.omit(dif[,k]), parallel = FALSE, verbose = FALSE))
          names(res_inner) <- colnames(dif)[k]
        }}
      res[i] <- res_inner
      names(res) <- task$feature_names[i]
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
  
  # If group CPI, rename groups
  if (!is.null(groups) & !is.null(names(groups))) {
    ret$Group <- names(groups)
  }
  
  # Reset to old logging threshold
  lgr::get_logger("mlr3")$set_threshold(old_logger_treshold)
  
  # Return CPI for all features/groups
  ret
}

compute_loss <- function(pred, measure, task, resampling = NULL, test_data = NULL) {
  if (inherits(pred, "Prediction")) {
    truth <- pred$truth
    response <- pred$response
    prob <- pred$prob
  } else if (inherits(pred[[1]], "PredictionSurv")) {
    prob.list = lapply(pred, function(x)
      as.data.frame(x$data$distr))
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
    loss <-
      graf_score(prob.list,
                 task,
                 resampling = resampling,
                 test_data = test_data)
  } else {
    stop("Unknown measure.")
  }
  loss
}
  
