####
# new function to compute the graf score for every element (n,t)
# input:
# prob.list: output of predict_learner
# task
# resampling 
# test data

# graf_score
graf_score <- function(prob.list, task, resampling = NULL, test_data = NULL) {
  ## create survival distribution matrix from (potential) multiple test sets
  unique.times <- unique(unlist(lapply(prob.list, names)))
  distr = do.call(rbind,
                  lapply(prob.list,
                         function(x)
                           data.frame(c(
                             x,
                             sapply(setdiff(unique.times,
                                            names(x)),
                                    function(y)
                                      NA)
                           ))))
  # create test set data.frame if resampling was and was not used
  data = task$data() # original dataset
  ids = NULL
  for (i in 1:length(prob.list)) {
    ids = append(ids, resampling$test_set(i)) 
  }
  test_data = data[ids,] 
  # convert data.frame to matrix and replace column names with predicted survival times
  distr.m = as.matrix(distr)
  pred.time = unlist(lapply(colnames(distr.m),
                            function(x)
                              as.numeric(gsub("X", "", x))))
  colnames(distr.m) = pred.time
  # sort matrix by row and column names
  distr.m <- distr.m[, order(as.numeric(colnames(distr.m)))]
  
  # use survfit to fit censoring distribution
  data$censor = 1 - data$status
  fit.km.censor = survfit(Surv(time, censor) ~ 1,
                          data = data,
                          type = c("kaplan-meier"))
  censor.prob = fit.km.censor$surv # estimated censoring probability at times t
  ## compute brier score for every (i,t) combination
  # create weight matrices from predicted censoring distribution
  censor.time = fit.km.censor$time
  weight.tstar = 1 / censor.prob[censor.time %in% pred.time]
  censor.dist = data.frame(cbind(censor.time, censor.prob))
  merge.data = merge(censor.dist, test_data, by.x = "censor.time", by.y = "time")
  weight.t = 1 / merge.data$censor.prob
  # create matrices for comparison in brier score
  tstar.m = matrix(rep(pred.time, each = nrow(distr.m)),
                   nrow = nrow(distr.m))
  t.m = matrix(
    rep(test_data$time, each = ncol(distr.m)),
    nrow = length(test_data$time),
    byrow = TRUE
  )
  censored.m = matrix(
    rep(test_data$status, each = ncol(distr.m)),
    nrow = length(test_data$time),
    byrow = TRUE
  )
  # binary matrices for comparison in brier score
  tlargertstar.m = ifelse(t.m > tstar.m, 1, 0)
  tsmallertstar.m = ifelse((t.m < tstar.m) &
                             (censored.m == 1) , 1, 0)
  #
  part1 = (distr.m ^ 2) * tsmallertstar.m
  part2 = ((1 - distr.m) ^ 2) * tlargertstar.m
  # brier score from two parts
  loss = apply(part1, 2, function(x)
    x * weight.t) +
    t(apply(part2, 1, function(x)
      x * weight.tstar))
  
  loss
}
