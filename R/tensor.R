AnvilTensor <- S7::new_S3_class("AnvilTensor")

#' @title Tensor
#' @description
#' Create a tensor.
#' @param data (any)\cr
#'   Object from which to create a tensor.
#'   Must be convertible to a [`PJRTBuffer`][pjrt::pjrt_buffer].
#' @param dtype (`NULL` | `character(1)` | [`stablehlo::TensorDataType`][data_types])\cr
#'   The type of the tensor.
#'   The default (`NULL`) uses `f32` for numeric data, `i32` for integer data, and `pred` for logical data.
#' @param platform (`NULL` | `character(1)`)\cr
#'   The platform name for the tensor (`"cpu"`, `"cuda"`, `"metal"`).
#' @param shape (`NULL` | `integer()`)\cr
#'   The shape of the tensor.
#'   The default (`NULL`) is to infer it from the data if possible.
#'   Note that [`nv_tensor`] interpretes length 1 vectors as having shape `(1)`.
#'   To create a "scalar" with dimension `()`, use [`nv_scalar`].
#' @details
#' Internally calls [`pjrt_buffer`][pjrt::pjrt_buffer].
#' @return (`AnvilTensor`)
#' @export
nv_tensor <- function(data, dtype = NULL, platform = NULL, shape = NULL) {
  x <- if (inherits(data, "PJRTBuffer")) {
    if (!is.null(platform) || !is.null(shape) || !is.null(dtype)) {
      stop(
        # fmt: skip
        "Cannot specify platform, shape, or dtype when data is already a PJRTBuffer. Use nv_convert() for type conversions." # nolint
      )
    }
    data
  } else {
    if (is_dtype(dtype)) {
      dtype <- as.character(dtype)
    }
    pjrt_buffer(data, dtype, client = platform %??% pjrt::pjrt_client("cpu"), shape)
  }
  ensure_nv_tensor(x)
}

ensure_nv_tensor <- function(x) {
  if (inherits(x, "AnvilTensor")) {
    return(x)
  }
  class(x) <- c("AnvilTensor", class(x))
  x
}

#' @rdname nv_tensor
#' @export
nv_scalar <- function(data, dtype = NULL, platform = "cpu") {
  x <- if (inherits(data, "PJRTBuffer")) {
    if (!is.null(platform) || !is.null(shape) || !is.null(dtype)) {
      stop(
        # fmt: skip
        "Cannot specify platform, shape, or dtype when data is already a PJRTBuffer. Use nv_convert() for type conversions." # nolint
      )
    }
    data
  } else {
    if (is_dtype(dtype)) {
      dtype <- as.character(dtype)
    }
    pjrt_scalar(data, dtype, client = platform %??% pjrt::pjrt_client("cpu"))
  }
  ensure_nv_tensor(x)
}

#' @rdname nv_tensor
#' @export
nv_empty <- function(dtype, shape, platform = "cpu") {
  if (is_dtype(dtype)) {
    dtype <- as.character(dtype)
  }
  x <- pjrt::pjrt_empty(dtype, shape, client = platform %??% pjrt::pjrt_client("cpu"))
  ensure_nv_tensor(x)
}

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

method(format, ShapedTensor) <- function(x, ...) {
  sprintf("ShapedTensor(dtype=%s, shape=%s)", repr(x@dtype), repr(x@shape))
}

method(print, ShapedTensor) <- function(x, ...) {
  cat(format(x), "\n")
}

method(print, ConcreteTensor) <- function(x, ...) {
  cat(format(x), "\n")
  print(x@data, header = FALSE)
}

method(format, ConcreteTensor) <- function(x, ...) {
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
