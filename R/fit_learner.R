
# Internal function to fit learner and compute prediction error
fit_learner <- function(learner, task, resampling = NULL, measure = NULL, test_data = NULL, verbose = FALSE) {
  if (!is.null(test_data)) {
    # Compute error on test data
    mod <- train(learner, task)
  } else if (is.list(resampling)) {
    # Full model resampling
    mod <- resample(learner, task, resampling = resampling, measures = measure, show.info = verbose, models = TRUE)$models
  } else if (resampling %in% c("none", "oob")) {
    # Compute error on training data
    mod <- train(learner, task)
  } else {
    stop("Unknown value for 'resampling'.")
  }
  mod
}

# Internal function to predict and compute prediction error
predict_learner <- function(mod, task, resampling = NULL, test_data = NULL) {
  if (!is.null(test_data)) {
    # Compute error on test data
    pred <- predict(mod, newdata = test_data)
  } else if (is.list(resampling)) {
    # Full model resampling
    pred <- lapply(seq_along(mod), function(i) {
      predict(mod[[i]], task, subset = resampling$test.inds[[i]])
    })
  } else if (resampling == "none") {
    # Compute error on training data
    pred <- predict(mod, task)
  } else if (resampling == "oob") {
    # Use OOB predictions if available
    learner_model <- getLearnerModel(mod)
    if (class(learner_model) == "ranger") {
      # ranger
      pred <- getOOBPreds(mod, task)
      if (is.null(learner_model$inbag.counts)) {
        stop("No inbag information available. Set 'keep.inbag = TRUE' in ranger.")
      } 
      preds <- predict(learner_model, getTaskData(task)[, getTaskFeatureNames(task)], predict.all = TRUE)$predictions
      oob_idx <- ifelse(simplify2array(learner_model$inbag.counts) == 0, TRUE, NA)
      if (length(dim(preds)) == 3) {
        # Probability forest
        for (i in 1:dim(preds)[2]) {
          preds[, i, ] <- oob_idx * preds[, i, ]
        }
        y_hat <- apply(preds, 1:2, mean, na.rm = TRUE)
        colnames(y_hat) <- learner_model$forest$levels[learner_model$forest$class.values]
        pred$data[, paste("prob", colnames(y_hat), sep = ".")] <- y_hat
      } else if (learner_model$treetype == "Classification") {
        apply(oob_idx * preds, 1, which.max)
        y_hat <- apply(oob_idx * preds, 1, function(x) {
          which.max(table(x, useNA = "no"))
        })
        y_hat <- learner_model$forest$levels[y_hat]
        y_hat <- factor(y_hat, levels = learner_model$forest$levels)
        pred$data$response <- y_hat
      } else {
        y_hat <- rowMeans(oob_idx * preds, na.rm = TRUE)
        pred$data$response <- y_hat
      }
    } else {
      stop("OOB error not available for this learner.")
    }
  } else {
    stop("Unknown value for 'resampling'.")
  }
  pred
}

