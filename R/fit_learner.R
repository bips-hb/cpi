
# Internal function to fit learner and compute prediction error
fit_learner <- function(learner, task, resampling = NULL, measure = NULL, test_data = NULL, verbose = FALSE) {
  if (!is.null(test_data)) {
    # Compute error on test data
    mod <- learner$train(task)
  } else if (inherits(resampling, "Resampling")) {
    # Full model resampling
    mod <- resample(task, learner, resampling, store_models = TRUE)
  } else if (is.character(resampling) && resampling %in% c("none", "oob")) {
    # Compute error on training data
    mod <- learner$train(task)
  } else {
    stop("Unknown value for 'resampling'.")
  }
  mod
}

# Internal function to predict and compute prediction error
predict_learner <- function(mod, task, resampling = NULL, test_data = NULL) {
  if (!is.null(test_data)) {
    # Compute error on test data
    pred <- mod$predict_newdata(test_data)
  } else if (inherits(resampling, "Resampling")) {
    # Full model resampling
    pred <- lapply(seq_along(mod$learners), function(i) {
      mod$learners[[i]]$predict(task, row_ids = resampling$test_set(i))
    })
  } else if (resampling == "none") {
    # Compute error on training data
    pred <- mod$predict_newdata(task$data())
  } else if (resampling == "oob") {
    # Use OOB predictions if available
    if (inherits(mod$model, "ranger")) {
      # ranger
      # In-sample predictions will be overriden below
      pred_data <- as.data.table(mod$predict_newdata(task$data()))
      if (is.null(mod$model$inbag.counts)) {
        stop("No inbag information available. Set 'keep.inbag = TRUE' in ranger.")
      }
      preds <- predict(mod$model, task$data(), predict.all = TRUE)$predictions
      oob_idx <- ifelse(simplify2array(mod$model$inbag.counts) == 0, TRUE, NA)
      if (length(dim(preds)) == 3) {
        # Probability forest
        for (i in 1:dim(preds)[2]) {
          preds[, i, ] <- oob_idx * preds[, i, ]
        }
        y_hat <- apply(preds, 1:2, mean, na.rm = TRUE)
        colnames(y_hat) <- mod$model$forest$levels[mod$model$forest$class.values]
        pred_data[, paste("prob", colnames(y_hat), sep = ".")] <- y_hat
        pred_data$response <- factor(colnames(y_hat)[max.col(y_hat)], 
                                     levels = levels(pred_data$response))
        pred <- as_prediction_classif(pred_data)
      } else if (mod$model$treetype == "Classification") {
        # Classification forest
        apply(oob_idx * preds, 1, which.max)
        y_hat <- apply(oob_idx * preds, 1, function(x) {
          which.max(table(x, useNA = "no"))
        })
        y_hat <- mod$model$forest$levels[y_hat]
        y_hat <- factor(y_hat, levels = mode$model$forest$levels)
        pred_data$response <- y_hat
        pred <- as_prediction_classif(pred_data)
      } else {
        # Regression forest
        y_hat <- rowMeans(oob_idx * preds, na.rm = TRUE)
        pred_data$response <- y_hat
        pred <- as_prediction_regr(pred_data)
      }
    } else {
      stop("OOB error not available for this learner.")
    }
  } else {
    stop("Unknown value for 'resampling'.")
  }
  pred
}

