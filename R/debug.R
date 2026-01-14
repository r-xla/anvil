#' @include tensor.R
#' @include box.R

#' @title Abstract Value Box Class
#' @description
#' Box representing a value in debug mode.
#' @param aval (`AbstractTensor`)\cr
#'   The abstract tensor representing the value.
#' @return (`DebugBox`)
#' @export
DebugBox <- function(aval) {
  checkmate::assert_class(aval, "AbstractTensor")

  structure(
    list(aval = aval),
    class = c("DebugBox", "AnvilBox")
  )
}

#' @export
shape.DebugBox <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.DebugBox <- function(x, ...) {
  dtype(x$aval)
}

#' @export
print.DebugBox <- function(x, ...) {
  aval <- x$aval
  if (is_concrete_tensor(aval)) {
    print(aval$data, ..., header = FALSE)
  } else if (is_literal_tensor(aval)) {
    cat(sprintf("%s:%s{%s}\n", aval$data, dtype2string(aval$dtype, aval$ambiguous), shape2string(aval$shape, FALSE))) # nolint
  } else {
    cat(sprintf("%s{%s}\n", dtype2string(aval$dtype, aval$ambiguous), shape2string(aval$shape, FALSE))) # nolint
  }
  invisible(x)
}

#' @title Create a Debug Box
#' @description
#' Create a debug box.
#' @template param_dtype
#' @template param_shape
#' @template param_ambiguous
#' @return ([`DebugBox`])
#' @export
debug_box <- function(dtype, shape, ambiguous = FALSE) {
  aval <- AbstractTensor(dtype = dtype, shape = shape, ambiguous = ambiguous)
  DebugBox(aval = aval)
}

is_debug_box <- function(x) {
  inherits(x, "DebugBox")
}
