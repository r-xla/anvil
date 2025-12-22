#' @title Assert Shape Vector
#' @description
#' Check whether an input is a valid shape vector (integer vector with all positive values).
#' @param x Object to check.
#' @param min.len (`integer(1)`)\cr
#'   Minimum length of the shape vector. Default is 1.
#' @param .var.name (`character(1)`)\cr
#'   Name of the variable to use in error messages.
#' @return Invisibly returns `x` if the assertion passes.
#' @keywords internal
assert_shapevec <- function(x, min.len = 1L, .var.name = rlang::caller_arg(x)) {
  ok <- test_integerish(x, lower = 1, min.len = min.len, any.missing = FALSE, null.ok = FALSE)
  if (!ok) {
    if (is.null(x) || !is.numeric(x)) {
      cli_abort("{.arg {.var.name}} must be an integer vector, not {.cls {class(x)}}")
    }
    if (anyNA(x)) {
      cli_abort("{.arg {.var.name}} must not contain missing values")
    }
    if (length(x) < min.len) {
      cli_abort("{.arg {.var.name}} must have at least {min.len} element{?s}")
    }
    if (any(x < 1)) {
      cli_abort("{.arg {.var.name}} must contain only positive integers (>= 1)")
    }
  }
  as.integer(x)
}
