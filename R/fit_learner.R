
# Internal function to fit learner and compute prediction error
fit_learner <- function(learner, task, resampling = NULL, measure = NULL, test_data = NULL, verbose = FALSE) {
  if (!is.null(test_data)) {
    # Compute error on test data
    mod <- train(learner, task)
    pred <- predict(mod, newdata = test_data)
  } else if (is.list(resampling)) {
    # Full model resampling
    r <- resample(learner, task, resampling = resampling, measures = measure, show.info = verbose, models = TRUE)
    pred <- r$pred
    mod <- r$models
  } else if (resampling == "none") {
    # Compute error on training data
    mod <- train(learner, task)
    pred <- predict(mod, task)
  } else if (resampling == "oob") {
    # Use OOB predictions if available
    if (!hasLearnerProperties(learner, "oobpreds")) {
      stop("OOB error not available for this learner.")
    }
    mod <- train(learner, task)
    pred <- getOOBPreds(mod, task)
  } else {
    stop("Unknown value for 'resampling'.")
  }
  list(pred = pred, mod = mod)
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
  } else {
    stop("Unknown value for 'resampling'.")
  }
  pred
}

