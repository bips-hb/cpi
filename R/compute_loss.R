
# Internal function to compute sample loss
compute_loss <- function(pred, measure) {
  if ("Prediction" %in% class(pred)) {
    pred_data <- pred$data
  } else {
    pred_data <- do.call(rbind, lapply(pred, function(x) x$data))
    pred <- pred[[1]]
  }
  truth <- pred_data$truth
  response <- pred_data$response
  
  if (getTaskType(pred) == "regr") {
    if (measure$id == "mse") {
      # Squared errors
      loss <- (truth - response)^2
    } else if (measure$id == "mae") {
      # Absolute errors
      loss <- abs(truth - response)
    } else {
      stop("Unknown measure.")
    }
  } else if (getTaskType(pred) == "classif") {
    if (measure$id == "logloss") {
      # Logloss 
      probabilities <- pred_data[, paste("prob", pred$task.desc$class.levels, sep = ".")]
      truth <- match(as.character(truth), pred$task.desc$class.levels)
      p <- probabilities[cbind(seq_len(nrow(probabilities)), truth)]
      loss <- -log(p)
    } else if (measure$id == "mmce") {
      # Misclassification error
      loss <- 1*(truth != response)
    } else if (measure$id == "brier") {
      # Brier score
      y <- as.numeric(truth == pred$task.desc$positive)
      loss <- (y - pred_data[, paste("prob", pred$task.desc$positive, sep = ".")])^2
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

