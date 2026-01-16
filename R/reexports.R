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

#' @importFrom tengen dtype
#' @export
tengen::dtype

#' @importFrom tengen ndims
#' @export
tengen::ndims

#' @importFrom stablehlo is_dtype
#' @export
stablehlo::is_dtype

#' @importFrom stablehlo as_dtype
#' @export
stablehlo::as_dtype

# FIXME(hack): https://github.com/sebffischer/S7-issue
#' @title Platform
#' @description
#' Get the platform of a tensor-like object.
#' @param x (any)\cr
#'   The tensor.
#' @param ... (`any`)\cr
#'   Additional argument (unused).
#' @return (`character(1)`)
#' @export
platform <- function(x, ...) {
  pjrt::platform(x)
}


#' @importFrom stablehlo Shape
#' @export
stablehlo::Shape

#' @importFrom stablehlo ScatterDimensionNumbers
#' @export
stablehlo::ScatterDimensionNumbers

#' @importFrom stablehlo GatherDimensionNumbers
#' @export
stablehlo::GatherDimensionNumbers
