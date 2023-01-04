
# Internal function to compute sample loss
compute_loss <- function(pred, measure) {
  if (inherits(pred, "Prediction")) {
    truth <- pred$truth
    response <- pred$response
    prob <- pred$prob
  } else {
    truth <- do.call(c, lapply(pred, function(x) x$truth))
    response <- do.call(c, lapply(pred, function(x) x$response))
    prob <- do.call(rbind, lapply(pred, function(x) x$prob))
  }
  
  if (measure$id == "regr.mse") {
    # Squared errors
    loss <- (truth - response)^2
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
    loss <- 1*(truth != response)
  } else if (measure$id == "classif.bbrier") {
    # Brier score
    # First level is positive class
    y <- as.numeric(as.numeric(truth) == 1)
    loss <- (y - prob[, 1])^2
  } else {
    stop("Unknown measure.")
  }
  
  loss
}

