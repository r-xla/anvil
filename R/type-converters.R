#' Convert AbstractTensor to ValueType
#' @description
#' Convert an `AbstractTensor` to a `ValueType`.
#' @param x (`AbstractTensor`)
#' @return (`ValueType`)
#' @export
at2vt <- function(x) {
  stopifnot(inherits(x, "AbstractTensor"))
  stablehlo::ValueType(stablehlo::TensorType(x$dtype, x$shape))
}

#' Convert ValueType to AbstractTensor
#' @description
#' Convert a `ValueType` to an `AbstractTensor`.
#' @param x (`ValueType`)
#' @return (`AbstractTensor`)
#' @export
vt2at <- function(x) {
  stopifnot(inherits(x, "ValueType"))
  stopifnot(inherits(x$type, "TensorType"))
  AbstractTensor(x$type$dtype, x$type$shape)
}