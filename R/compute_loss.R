
# Internal function to compute sample loss
compute_loss <- function(pred, measure) {
  if ("Prediction" %in% class(pred)) {
    pred_data <- pred$data
  } else {
    if (getTaskType(pred[[1]]) == "classif" & measure$id == "logloss") {
      # Assure same order for classes
      pred_data <- do.call(rbind, lapply(pred, function(x) x$data[, c("id", "truth", paste("prob", x$task.desc$class.levels, sep = "."), "response")]))
    } else {
      pred_data <- do.call(rbind, lapply(pred, function(x) x$data))
    }
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
      p[p < 1e-15] <- 1e-15 # Avoid infinity
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
  } else {
    stop("Unknown task type.")
  }
  
  loss
}

