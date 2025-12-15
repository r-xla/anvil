#' @rdname nv_tensor
AnvilTensor <- S7::new_S3_class("AnvilTensor")

#' @importFrom pjrt platform

#' @title Tensor
#' @description
#' Create a tensor.
#' @param data (any)\cr
#'   Object convertible to a [`PJRTBuffer`][pjrt::pjrt_buffer].
#' @param dtype (`NULL` | `character(1)` | [`TensorDataType`])\cr
#'   One of `r stablehlo:::roxy_dtypes()` or a [`stablehlo::TensorDataType`].
#'   The default (`NULL`) uses `f32` for numeric data, `i32` for integer data, and `pred` for logical data.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The platform name for the tensor (`"cpu"`, `"cuda"`, `"metal"`).
#'   Default is to use the CPU, unless the data is already a [`PJRTBuffer`][pjrt::pjrt_buffer].
#'   You can change the default by setting the `PJRT_PLATFORM` environment variable.
#' @param shape (`NULL` | `integer()`)\cr
#'   Shape.
#'   The default (`NULL`) is to infer it from the data if possible.
#'   Note that [`nv_tensor`] interprets length 1 vectors as having shape `(1)`.
#'   To create a "scalar" with dimension `()`, use [`nv_scalar`].
#' @details
#' Internally calls [`pjrt_buffer`][pjrt::pjrt_buffer].
#' @return (`AnvilTensor`)
#' @export
nv_tensor <- function(data, dtype = NULL, device = NULL, shape = NULL) {
  if (is_dtype(dtype)) {
    dtype <- as.character(dtype)
  }
  x <- pjrt_buffer(data, dtype, device = device, shape = shape)
  ensure_nv_tensor(x)
}

is_anvil_tensor <- function(x) {
  inherits(x, "AnvilTensor")
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
nv_scalar <- function(data, dtype = NULL, device = NULL) {
  if (is_dtype(dtype)) {
    dtype <- as.character(dtype)
  }
  x <- pjrt_scalar(data, dtype, device = device)
  ensure_nv_tensor(x)
}

#' @rdname nv_tensor
#' @export
nv_empty <- function(dtype, shape, device = NULL) {
  if (is_dtype(dtype)) {
    dtype <- as.character(dtype)
  }
  x <- pjrt::pjrt_empty(dtype, shape, device = device)
  ensure_nv_tensor(x)
}

#' @export
dtype.AnvilTensor <- function(x, ...) {
  as_dtype(as.character(pjrt::elt_type(x)))
}

#' @title Shaped Tensor Class
#' @description
#' Abstract representation of a tensor with a known dtype and shape, but no concrete data.
#' Used during tracing to represent tensor metadata without actual values.
#'
#' @param dtype ([`stablehlo::TensorDataType`])\cr
#'   The data type of the tensor.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the tensor. Can be provided as an integer vector.
#' @param ambiguous (`logical(1)`)\cr
#'   Whether the type is ambiguous. Ambiguous usually arise from R literals
#'   (e.g., `1L`, `1.0`, `TRUE`) and follow special promotion rules.
#'   Only `f32`, `i32`, and `i1` (bool) can be ambiguous.
#' @seealso [ConcreteTensor], [LiteralTensor], [st()]
#' @export
ShapedTensor <- S7::new_class(
  "ShapedTensor",
  properties = list(
    dtype = stablehlo::TensorDataType,
    shape = stablehlo::Shape,
    # ambiguous types are literals or values that were cast to an ambiguous type
    ambiguous = new_property(class_logical, validator = function(value) {
      if (!test_flag(value)) {
        return("ambiguous must be a flag")
      }
    })
  ),
  constructor = function(dtype, shape, ambiguous = FALSE) {
    shape <- as_shape(shape)
    dtype <- as_dtype(dtype)
    if (ambiguous) {
      ok <- is_dtype(dtype) && (dtype == dtype("f32") || dtype == dtype("i32") || dtype == dtype("i1"))
      if (!ok) {
        cli_abort("Ambiguous types must have dtype f32, i32 or bool")
      }
    }
    S7::new_object(
      S7::S7_object(),
      dtype = dtype,
      shape = shape,
      ambiguous = ambiguous
    )
  }
)

shaped_tensor <- function(x) {
  if (is_anvil_tensor(x)) {
    ConcreteTensor(x)
  } else if (is_shaped_tensor(x)) {
    x
  } else {
    cli_abort("internal error")
  }
}


is_shaped_tensor <- function(x) {
  inherits(x, "anvil::ShapedTensor")
}

is_concrete_tensor <- function(x) {
  inherits(x, "anvil::ConcreteTensor")
}

method(platform, ShapedTensor) <- function(x, ...) {
  cli_abort("platform is not accessible during tracing")
}

#' @method dtype anvil::ShapedTensor
#' @export
`dtype.anvil::ShapedTensor` <- function(x, ...) {
  x@dtype
}

#' @method shape anvil::ShapedTensor
#' @export
`shape.anvil::ShapedTensor` <- function(x, ...) {
  x@shape@dims
}

#' @title Concrete Tensor Class
#' @description
#' A [`ShapedTensor`] that also holds a reference to the actual tensor data.
#' Used to represent constants captured during tracing.
#' Because it comes from a concrete tensor, it's type is not ambiguous.
#'
#' @param data ([`AnvilTensor`])\cr
#'   The actual tensor data.
#' @seealso [ShapedTensor], [LiteralTensor]
#' @export
ConcreteTensor <- S7::new_class(
  "ConcreteTensor",
  parent = ShapedTensor,
  properties = list(
    data = AnvilTensor
  ),
  constructor = function(data) {
    if (!inherits(data, "AnvilTensor")) {
      cli_abort("data must be an AnvilTensor")
    }

    S7::new_object(
      S7::S7_object(),
      dtype = dtype_from_buffer(data),
      shape = Shape(shape(data)),
      data = data,
      ambiguous = FALSE
    )
  }
)

#' @title Literal Tensor Class
#' @description
#' A [`ShapedTensor`] representing a tensor where the data is a R scalar literal (e.g., `1L`, `2.5`, `TRUE`).
#' Their type is ambiguous, and they adapt (if possible) to the types of
#' non-literal tensors they interact with.
#'
#' @param data (`numeric(1)` | `integer(1)` | `logical(1)`)\cr
#'   The scalar value.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the tensor.
#' @param dtype ([`stablehlo::TensorDataType`])\cr
#'   The data type. Defaults to `f32` for numeric, `i32` for integer, `i1` for logical.
#' @seealso [ShapedTensor], [ConcreteTensor]
#' @export
LiteralTensor <- new_class(
  "LiteralTensor",
  parent = ShapedTensor,
  properties = list(
    data = new_property(class_any, validator = function(value) {
      if (!test_scalar(value)) {
        return("LiteralTensors expect scalars")
      }
    }),
    shape = stablehlo::Shape,
    dtype = stablehlo::TensorDataType
  ),
  constructor = function(data, shape, dtype = default_dtype(data)) {
    if (!test_scalar(data)) {
      cli_abort("LiteralTensors expect scalars")
    }
    if (is.numeric(shape)) {
      shape <- Shape(as.integer(shape))
    }
    S7::new_object(
      S7::S7_object(),
      data = data,
      shape = shape,
      dtype = default_dtype(data),
      ambiguous = TRUE
    )
  }
)

is_literal_tensor <- function(x) {
  inherits(x, "anvil::LiteralTensor")
}

method(platform, ConcreteTensor) <- function(x, ...) {
  pjrt::platform(x@data)
}

method(`==`, list(ShapedTensor, ShapedTensor)) <- function(e1, e2) {
  e1@dtype == e2@dtype && e1@shape == e2@shape && e1@ambiguous == e2@ambiguous
}

method(repr, ShapedTensor) <- function(x) {
  sprintf("%s[%s]", paste0(repr(x@dtype), if (x@ambiguous) "?"), repr(x@shape))
}

method(format, ShapedTensor) <- function(x, ...) {
  sprintf(
    "ShapedTensor(dtype=%s, shape=%s)",
    if (x@ambiguous) paste0(repr(x@dtype), "?") else repr(x@dtype),
    repr(x@shape)
  ) # nolint
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


#' @export
format.AnvilTensor <- function(x, ...) {
  sprintf("AnvilTensor(dtype=%s, shape=%s)", repr(dtype(x)), paste(shape(x), collapse = "x"))
}

#' @title Convert to Shaped Tensor
#' @description
#' Convert an object to its abstract tensor representation ([`ShapedTensor`]).
#' @param x (`any`)\cr
#'   Object to convert.
#' @return [`ShapedTensor`]
#' @export
st <- function(x) {
  if (is_anvil_tensor(x)) {
    ConcreteTensor(x)
  } else if (is_shaped_tensor(x)) {
    x
  } else if (test_atomic(x) && (is.logical(x) || is.numeric(x))) {
    LiteralTensor(x, integer())
  } else if (is_graph_box(x)) {
    gnode <- x@gnode
    if (is_graph_value(gnode)) {
      gnode@aval
    } else {
      st(gnode)
    }
  } else {
    cli_abort("internal error")
  }
}

#' @export
dtype.character <- function(x, ...) {
  as_dtype(x)
}

as_shape <- function(x) {
  if (test_integerish(x, any.missing = FALSE, lower = 0)) {
    Shape(as.integer(x))
  } else if (is_shape(x)) {
    x
  } else if (is.null(x)) {
    Shape(integer())
  } else {
    cli_abort("x must be an integer vector or a stablehlo::Shape")
  }
}

is_shape <- function(x) {
  inherits(x, "stablehlo::Shape")
}
