#' @title Tensor
#' @description
#' Create a tensor.
#' @param x (any)\cr
#'   Object.
#' @param ... (any)\cr
#'   Additional arguments.
#' @details
#' Internally calls [`pjrt_buffer`][pjrt::pjrt_buffer].
#' @return `nv_tensor`
#' @export
nv_tensor <- S7::new_generic("AnvilTensor", "x", function(x, ...) {
  S7::S7_dispatch()
})

method(nv_tensor, S7::new_S3_class("PJRTBuffer")) <- function(x, ...) {
  class(x) <- c("AnvilTensor", "PJRTBuffer")
  x
}

method(nv_tensor, class_any) <- function(x, ...) {
  structure(pjrt::pjrt_buffer(x, ...), class = c("AnvilTensor", "PJRTBuffer"))
}

#' @export
nv_scalar <- S7::new_generic("nv_scalar", "x", function(x, ...) {
  S7::S7_dispatch()
})

method(nv_scalar, S7::new_S3_class("PJRTBuffer")) <- function(x, ...) {
  class(x) <- c("AnvilTensor", "PJRTBuffer")
  x
}

method(nv_scalar, class_any) <- function(x, ...) {
  structure(pjrt::pjrt_scalar(x, ...), class = c("AnvilTensor", "PJRTBuffer"))
}

AnvilTensor <- S7::new_S3_class("AnvilTensor")

method(dtype, AnvilTensor) <- function(x) {
  as_dtype(as.character(pjrt::elt_type(x)))
}

# similar to stablehlo::TensorType
ShapedTensor <- S7::new_class(
  "ShapedTensor",
  properties = list(
    dtype = stablehlo::TensorDataType,
    shape = stablehlo::Shape
  )
)

method(dtype, ShapedTensor) <- function(x) {
  x@dtype
}

method(shape, ShapedTensor) <- function(x) {
  x@shape@dims
}

# Used primarily for constants
ConcreteTensor <- S7::new_class(
  "ConcreteTensor",
  parent = ShapedTensor,
  properties = list(
    data = AnvilTensor
  ),
  constructor = function(data) {
    if (!inherits(data, "AnvilTensor")) {
      stop("data must be an nv_tensor")
    }

    S7::new_object(
      S7::S7_object(),
      dtype = dtype_from_buffer(data),
      shape = Shape(shape(data)),
      data = data
    )
  }
)

method(`==`, list(ShapedTensor, ShapedTensor)) <- function(e1, e2) {
  e1@dtype == e2@dtype && e1@shape == e2@shape
}

method(repr, ShapedTensor) <- function(x) {
  sprintf("%s[%s]", repr(x@dtype), repr(x@shape))
}

method(format, ShapedTensor) <- function(x) {
  sprintf("ShapedTensor(dtype=%s, shape=%s)", repr(x@dtype), repr(x@shape))
}

method(print, ShapedTensor) <- function(x) {
  cat(format(x), "\n")
}

method(print, ConcreteTensor) <- function(x) {
  cat(format(x), "\n")
  print(x@data, header = FALSE)
}

method(format, ConcreteTensor) <- function(x) {
  sprintf("ConcreteTensor(dtype=%s, shape=%s)", repr(x@dtype), repr(x@shape))
}


#' @title Create a TensorDataType
#' @description
#' Create a [`stablehlo::TensorDataType`].
#' @param x (any)\cr
#'   Object convertible to a [`stablehlo::TensorDataType`] (via [`stablehlo::as_dtype`])
#' @return [`stablehlo::TensorDataType`]
#' @export
nv_dtype <- function(x) {
  stablehlo::as_dtype(x)
}
