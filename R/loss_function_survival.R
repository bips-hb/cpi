# Internal functions to to compute survival loss based on internal mlr3 functions

#'  Weighted survival score
#'  Internal Function of mlr3 package to compute survival loss for every (i,t) combination
#' in passed vector of predicted survival distribution, slightly adjusted
#' Arguments:
#'  @param loss: type of survival loss to be computed ("schmid", "graf", "logloss")
#'  @param truth: matrix with two columns, column 1: survival times corresponding to every i,
#'  column 2 censoring indicator of every i
#'  @param distribution: predicted survival distribution, either discrete or continuous
#'  @param times: vector of times, if specified survival loss will be computed for
#'  pred: outcome of predict_leaner function
#'  @param t_max: if specified survival loss will be computed for seq(t_max)
#'  @param p_max:proportion of censoring to integrate up to in the dataset (https://mlr3proba.mlr-org.com/news/index.html) 
#'  @param proper: determines what weighting scheme should be applied by the estimated censoring distribution
#'  The current method (Graf, 1999) proper = FALSE, weights observations either by 
#'  their event time or ‘current’ time depending if they’re dead or not, the new method 
#'  @param proper = TRUE weights observations by event time. (https://mlr3proba.mlr-org.com/news/index.html)
#'  @param train = NULL: 
#'  @param eps: parameter added to control bug if probability of censoring is 0 (https://mlr3proba.mlr-org.com/news/index.html)

#'  t_max argument not null -> obtain errors for all times seq(t_max), can be used with predefined test_data and resampling of any kind
#'  times argument not null -> obtain errors for defined vector of times, can be used with predefined test_data and resamplinhg of any kind
#'  both t_max and times argument null -> by default use all times in test set to obtain errors, can NOT be used if resampling with multiple folds is used


weighted_survival_score = function(loss, truth, distribution, times, pred, 
                                   t_max, p_max, proper, train = NULL, eps, ...) {
  
  assert_surv(truth)
  
# 
#  if (is.null(times) || !length(times)) {
#    unique_times = unique(sort(truth[, "time"]))
#    if (!is.null(t_max)) {
#      #     unique_times  = unique_times[unique_times <= t_max]
#      unique_times  = seq(t_max)
#    } else if (!is.null(p_max)) {
#      s = survival::survfit(truth ~ 1)
#      t_max = s$time[which(1 - s$n.risk / s$n > p_max)[1]]
#      unique_times  = unique_times[unique_times <= t_max]
#    }
#  } else {
#    unique_times = .c_get_unique_times(truth[, "time"], times)
#  }
  
    if (!is.null(times) & is.null(t_max)) {
      unique_times = unique(sort(times))
    } else if (is.null(times) & !is.null(t_max)) {
        unique_times  = seq(t_max)
    } else if (is.null(times) & is.null(t_max)) {
        unique_times  = seq(min(truth[, "time"]), max(truth[, "time"]), length.out = 100)
#      if (length(pred) == 1) {
#        unique_times = .c_get_unique_times(truth[, "time"], truth[, "time"])
#      } else {
#        stop("Default times cannot be used with multiple folds. Please specify either 'times' or 't_max' argument.")  
#        }
    } else {
      stop("'times' and 't_max' argument cannot be non-zero at the same time.")
    }
  
  if (inherits(distribution, "Distribution")) {
    cdf = as.matrix(distribution$cdf(unique_times))
  } else {
    mtc = findInterval(unique_times, as.numeric(colnames(distribution)))
    cdf = 1 - t(distribution[, mtc])
    if (any(mtc == 0)) {
      cdf = rbind(matrix(0, sum(mtc == 0), ncol(cdf)), cdf)
    }
    rownames(cdf) = unique_times
  }
  
  true_times <- truth[, "time"]
  
  assert_numeric(true_times, any.missing = FALSE)
  
  assert_numeric(unique_times, any.missing = FALSE)
  assert_matrix(cdf, nrows = length(unique_times), ncols = length(true_times), any.missing = FALSE)
  
  ## Note that whilst we calculate the score for censored here, they are then
  ##  corrected in the weighting function
  if (loss == "graf") {
    score = score_graf_schmid(true_times, unique_times, cdf, power = 2)
  } else if (loss == "schmid") {
    score = score_graf_schmid(true_times, unique_times, cdf, power = 1)
  } else {
    score = score_intslogloss(true_times, unique_times, cdf, eps = eps)
  }
  
  if (is.null(train)) {
    cens = survival::survfit(Surv(truth[, "time"], 1 - truth[, "status"]) ~ 1)
  } else {
    cens = survival::survfit(Surv(train[, "time"], 1 - train[, "status"]) ~ 1)
  }
  
  score = .c_weight_survival_score(score, truth, unique_times, matrix(c(cens$time, cens$surv), ncol = 2), proper, eps)
  colnames(score) = unique_times
  
  return(score)
}


# Internal mlr3 functions necessary to compute survival loss 

.c_weight_survival_score <- function(score, truth, unique_times, cens, proper, eps) {
  .Call("_mlr3proba_c_weight_survival_score", score, truth, unique_times, cens, proper, eps)
}

.c_get_unique_times <- function(true_times, req_times) {
  .Call("_mlr3proba_c_get_unique_times", true_times, req_times)
}

score_graf_schmid = function(true_times, unique_times, cdf, power = 2) {
  assert_number(power)
  c_score_graf_schmid(true_times, unique_times, cdf, power)
}

c_score_graf_schmid <- function(truth, unique_times, cdf, power = 2L) {
  .Call("_mlr3proba_c_score_graf_schmid", truth, unique_times, cdf, power)
}

c_score_intslogloss <- function(truth, unique_times, cdf, eps) {
  .Call("_mlr3proba_c_score_intslogloss", truth, unique_times, cdf, eps)
}

score_intslogloss = function(true_times, unique_times, cdf, eps = eps) {
  assert_number(eps, lower = 0)
  c_score_intslogloss(true_times, unique_times, cdf, eps = eps)
}


## Survival Loss
## Function that computes survival score for every (i,t) combination with i's taken 
# from every test set fold, outputs matrix of (i,t) scores
# Arguments:
# @param loss: type of survival loss to be computed ("schmid", "graf", "logloss")
# @param pred: outcome of predict_leaner function
# @param t_max: if specified survival loss will be computed for seq(t_max)
# @param times: vector of times, if specified survival loss will be computed for
# @param resampling: mlr3 resampling object if it exists
# @param test_data: test data dataframe if it exists

survival_loss <- function(loss, pred, t_max, times, resampling, test_data){
  
  # Set default values for arguments required by mlr3 internal functions
  p_max = NULL
  eps = 1e-3 
  proper = FALSE 
  train = NULL 

  # Compute weighted survival score for each (i,t) combination for every test fold 
  full_score_list = vector("list", length(pred))
  # Loop over test folds
  for (i in 1:length(pred)) {
    truth = pred[[i]]$truth
    distribution = pred[[i]]$data$distr
    score = weighted_survival_score(loss = loss,
                                    truth = truth,
                                    distribution = distribution, 
                                    times = times,
                                    pred = pred,
                                    t_max = t_max, 
                                    p_max = p_max, 
                                    proper = proper, 
                                    train = train,
                                    eps = eps)
    if (!is.null(resampling)){
      rownames(score) = resampling$test_set(i)
    } else {
      rownames(score) = rownames(test_data) ### do rownames of test_data dataset really correspond to ids?
    }
    full_score_list[[i]] = score
  }

  # Combine weighted survival scores of every fold
  full_score = do.call(rbind, full_score_list)
  return(full_score)
}


