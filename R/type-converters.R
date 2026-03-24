#' Convert AbstractArray to ValueType
#' @description
#' Convert an `AbstractArray` to a `ValueType`.
#' @param x (`AbstractArray`)
#' @return (`ValueType`)
#' @export
at2vt <- function(x) {
  stopifnot(inherits(x, "AbstractArray"))
  stablehlo::ValueType(stablehlo::TensorType(x$dtype, x$shape))
}

#' Convert ValueType to AbstractArray
#' @description
#' Convert a `ValueType` to an `AbstractArray`.
#' @param x (`ValueType`)
#' @return (`AbstractArray`)
#' @export
vt2at <- function(x) {
  stopifnot(inherits(x, "ValueType"))
  stopifnot(inherits(x$type, "TensorType"))
  AbstractArray(x$type$dtype, x$type$shape)
}
