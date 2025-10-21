#' @importFrom tengen shape
#' @export
tengen::shape

#' @importFrom tengen device
#' @export
tengen::device

#' @importFrom tengen as_array
#' @export
tengen::as_array

#' @importFrom tengen as_raw
#' @export
tengen::as_raw

#' @importFrom tengen ndims
#' @export
tengen::ndims

#' @importFrom stablehlo is_dtype
#' @export
stablehlo::is_dtype

#' @include interpreter.R
# FIXME(hack): https://github.com/sebffischer/S7-issue
#' @title Platform
#' @description
#' Get the platform of a tensor.
#' @param x (`anvil::Box`)\cr
#'   The tensor.
#' @param ... (`any`)\cr
#'   Additional argument (unused).
#' @return (`character(1)`)\cr
#'   The platform of the tensor.
#' @export
platform <- function(x, ...) {
  pjrt::platform(aval(x))
}
