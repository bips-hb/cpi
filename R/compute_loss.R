
# Internal function to compute sample loss
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

