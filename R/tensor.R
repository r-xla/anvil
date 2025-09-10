#' @title Tensor
#' @description
#' Create an tensor.
#' @param x (any)\cr
#'   Object.
#' @param ... (any)\cr
#'   Additional arguments.
#' @details
#' Internally calls [`pjrt_buffer`][pjrt::pjrt_buffer].
#' @return `nvl_tensor`
#' @export
nvl_tensor <- function(x, ...) {
  structure(pjrt::pjrt_buffer(x, ...), class = c("nvl_tensor", "PJRTBuffer"))
}

#' @export
nvl_scalar <- function(x, ...) {
  structure(pjrt::pjrt_scalar(x, ...), class = c("nvl_tensor", "PJRTBuffer"))
}

#' @export
print.nvl_tensor <- function(x, header = TRUE, ...) {
  if (assert_flag(header)) {
    s <- shape(x)
    cat(
      sprintf(
        "nvl_tensor<%s%s>",
        as.character(dtype(x)),
        if (length(s)) paste0(": ", paste0(s, collapse = "x"))
      ),
      "\n"
    )
  }
  NextMethod("print", header = FALSE, ...)
}

# TODO: maybe rename to NvlTensor
Tensor <- S7::new_S3_class("nvl_tensor")

# similar to stablehlo::TensorType
ShapedTensor <- S7::new_class(
  "ShapedTensor",
  properties = list(
    dtype = stablehlo::TensorElementType,
    shape = stablehlo::Shape
  )
)

# Used primarily for constants
ConcreteTensor <- S7::new_class(
  "ConcreteTensor",
  parent = ShapedTensor,
  properties = list(
    data = Tensor
  ),
  constructor = function(data) {
    if (!inherits(data, "nvl_tensor")) {
      stop("data must be an nvl_tensor")
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
