
# Function to concatenate factors not defined in R<4.1.0
if (!exists("c.factor")) {
  c.factor <- function (..., recursive = TRUE) {
    x <- list(...)
    y <- unlist(x, recursive = recursive)
    if (inherits(y, "factor") && all(vapply(x, inherits, NA, 
        "ordered")) && (length(unique(lapply(x, levels))) == 1L)) 
      class(y) <- c("ordered", "factor")
    y
  }
}