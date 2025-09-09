#' @title Array
#' @description
#' Create an array.
#' @param x (any)\cr
#'   Object.
#' @param ... (any)\cr
#'   Additional arguments.
#' @details
#' Internally calls [`pjrt_buffer`][pjrt::pjrt_buffer].
#' @return `nvl_array`
#' @export
nvl_array <- function(x, ...) {
  structure(pjrt::pjrt_buffer(x, ...), class = c("nvl_array", "PJRTBuffer"))
}

#' @export
nvl_scalar <- function(x, ...) {
  structure(pjrt::pjrt_scalar(x, ...), class = c("nvl_array", "PJRTBuffer"))
}

#' @export
print.nvl_array <- function(x, header = TRUE, ...) {
  if (assert_flag(header)) {
    s <- shape(x)
    cat(
      sprintf(
        "nvl_array<%s%s>",
        as.character(dtype(x)),
        if (length(s)) paste0(": ", paste0(s, collapse = "x"))
      ),
      "\n"
    )
  }
  NextMethod("print", header = FALSE, ...)
}

# TODO: maybe rename to NvlArray
Array <- S7::new_S3_class("nvl_array")

# similar to stablehlo::TensorType
ShapedArray <- S7::new_class(
  "ShapedArray",
  properties = list(
    dtype = stablehlo::TensorElementType,
    shape = stablehlo::Shape
  )
)

# Used primarily for constants
ConcreteArray <- S7::new_class(
  "ConcreteArray",
  parent = ShapedArray,
  properties = list(
    data = Array
  ),
  constructor = function(data) {
    if (!inherits(data, "nvl_array")) {
      stop("data must be an nvl_array")
    }

    S7::new_object(
      S7::S7_object(),
      dtype = dtype_from_buffer(data),
      shape = Shape(shape(data)),
      data = data
    )
  }
)

method(`==`, list(ShapedArray, ShapedArray)) <- function(e1, e2) {
  e1@dtype == e2@dtype && e1@shape == e2@shape
}

method(repr, ShapedArray) <- function(x) {
  sprintf("%s[%s]", repr(x@dtype), repr(x@shape))
}

method(format, ShapedArray) <- function(x) {
  sprintf("ShapedArray(dtype=%s, shape=%s)", repr(x@dtype), repr(x@shape))
}

method(print, ShapedArray) <- function(x) {
  cat(format(x), "\n")
}

method(print, ConcreteArray) <- function(x) {
  cat(format(x), "\n")
  print(x@data, header = FALSE)
}

method(format, ConcreteArray) <- function(x) {
  sprintf("ConcreteArray(dtype=%s, shape=%s)", repr(x@dtype), repr(x@shape))
}
