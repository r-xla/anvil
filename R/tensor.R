#' @title AnvilTensor
#' @description
#' Virtual base class for tensor objects in anvil.
#' This class is used to mark objects that can be used as tensors in anvil.
#' Cannot be instantiated directly - use [`nv_tensor()`], [`nv_scalar()`], or [`nv_empty()`] instead.
#' @name AnvilTensor
NULL

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

#' @rdname AbstractTensor
#' @export
nv_aten <- function(dtype, shape, ambiguous = FALSE) {
  AbstractTensor(dtype = dtype, shape = shape, ambiguous = ambiguous)
}

#' @export
dtype.AnvilTensor <- function(x, ...) {
  as_dtype(as.character(pjrt::elt_type(x)))
}

#' @title Abstract Tensor Class
#' @description
#' Abstract representation of a tensor with a (possibly ambiguous) dtype and shape, but no concrete data.
#' Used during tracing to represent tensor metadata without actual values.
#'
#' @details
#' Two tensors are considered equal (`==`) if they have the same dtype and shape, ignoring ambiguity.
#'
#' @param dtype ([`stablehlo::TensorDataType`])\cr
#'   The data type of the tensor.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the tensor. Can be provided as an integer vector.
#' @template param_ambiguous
#' @seealso [ConcreteTensor], [LiteralTensor], [to_abstract()]
#' @export
AbstractTensor <- function(dtype, shape, ambiguous = FALSE) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  if (!test_flag(ambiguous)) {
    cli_abort("ambiguous must be a flag")
  }
  if (ambiguous) {
    ok <- is_dtype(dtype) && (repr(dtype) == "f32" || repr(dtype) == "i32")
    if (!ok) {
      cli_abort("Ambiguous types must have dtype f32 or i32")
    }
  }

  structure(
    list(dtype = dtype, shape = shape, ambiguous = ambiguous),
    class = "AbstractTensor"
  )
}

is_abstract_tensor <- function(x) {
  inherits(x, "AbstractTensor")
}

is_concrete_tensor <- function(x) {
  inherits(x, "ConcreteTensor")
}

#' @title Platform for AbstractTensor
#' @description
#' Get the platform of an AbstractTensor. Always errors since platform
#' is not accessible during tracing.
#' @param x An AbstractTensor.
#' @param ... Additional arguments (unused).
#' @return Never returns; always errors.
#' @export
platform.AbstractTensor <- function(x, ...) {
  cli_abort("platform is not accessible during tracing")
}

#' @method dtype AbstractTensor
#' @export
dtype.AbstractTensor <- function(x, ...) {
  x$dtype
}

#' @method shape AbstractTensor
#' @export
shape.AbstractTensor <- function(x, ...) {
  x$shape$dims
}

#' @title Concrete Tensor Class
#' @description
#' A [`AbstractTensor`] that also holds a reference to the actual tensor data.
#' Used to represent constants captured during tracing.
#' Because it comes from a concrete tensor, it's type is never ambiguous.
#'
#' @param data ([`AnvilTensor`])\cr
#'   The actual tensor data.
#' @seealso [AbstractTensor], [LiteralTensor]
#' @export
ConcreteTensor <- function(data) {
  if (!inherits(data, "AnvilTensor")) {
    cli_abort("data must be an AnvilTensor")
  }

  structure(
    list(
      dtype = dtype_from_buffer(data),
      shape = Shape(shape(data)),
      data = data,
      ambiguous = FALSE
    ),
    class = c("ConcreteTensor", "AbstractTensor")
  )
}

#' @title Literal Tensor Class
#' @description
#' A [`AbstractTensor`] representing a tensor where the data is a R scalar literal (e.g., `1L`, `2.5`)
#' or an [`AnvilTensor`].
#' Usually, their type is ambiguous, unless created via [`nv_fill`].
#'
#' @param data (`numeric(1)` | `integer(1)` | `logical(1)` | [`AnvilTensor`])\cr
#'   The scalar value or scalarish AnvilTensor (contains 1 element).
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the tensor.
#' @param dtype ([`stablehlo::TensorDataType`])\cr
#'   The data type. Defaults to `f32` for numeric, `i32` for integer, `i1` for logical.
#' @template param_ambiguous
#' @seealso [AbstractTensor], [ConcreteTensor]
#' @export
LiteralTensor <- function(data, shape, dtype = default_dtype(data), ambiguous) {
  if (!test_scalar(data) && !inherits(data, "AnvilTensor")) {
    cli_abort("LiteralTensors expect scalars or AnvilTensor")
  }
  if (inherits(data, "AnvilTensor")) {
    if (prod(shape(data)) != 1L) {
      cli_abort("AnvilTensor must contain exactly one element.")
    }
  }
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)

  structure(
    list(
      data = data,
      dtype = dtype,
      shape = shape,
      ambiguous = ambiguous
    ),
    class = c("LiteralTensor", "AbstractTensor")
  )
}

is_literal_tensor <- function(x) {
  inherits(x, "LiteralTensor")
}

#' @title Platform for ConcreteTensor
#' @description
#' Get the platform of a ConcreteTensor.
#' @param x A ConcreteTensor.
#' @param ... Additional arguments (unused).
#' @return The platform string.
#' @export
platform.ConcreteTensor <- function(x, ...) {
  pjrt::platform(x$data)
}

#' @export
`==.AbstractTensor` <- function(e1, e2) {
  if (!inherits(e2, "AbstractTensor")) {
    return(FALSE)
  }
  e1$dtype == e2$dtype && e1$shape == e2$shape
}

#' @export
`!=.AbstractTensor` <- function(e1, e2) {
  !(e1 == e2)
}

#' @export
repr.AbstractTensor <- function(x, ...) {
  sprintf("%s[%s]", paste0(repr(x$dtype), if (x$ambiguous) "?"), repr(x$shape))
}

#' @export
format.AbstractTensor <- function(x, ...) {
  sprintf(
    "AbstractTensor(dtype=%s, shape=%s)",
    if (x$ambiguous) paste0(repr(x$dtype), "?") else repr(x$dtype),
    repr(x$shape)
  )
}

#' @export
print.AbstractTensor <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
print.ConcreteTensor <- function(x, ...) {
  cat(format(x), "\n")
  print(x$data, header = FALSE)
  invisible(x)
}

#' @export
format.ConcreteTensor <- function(x, ...) {
  sprintf("ConcreteTensor(dtype=%s, shape=%s)", repr(x$dtype), repr(x$shape))
}


#' @export
format.AnvilTensor <- function(x, ...) {
  sprintf("AnvilTensor(dtype=%s, shape=%s)", repr(dtype(x)), paste(shape(x), collapse = "x"))
}

#' @title Convert to Abstract Tensor
#' @description
#' Convert an object to its abstract tensor representation ([`AbstractTensor`]).
#' @param x (`any`)\cr
#'   Object to convert.
#' @param pure (`logical(1)`)\cr
#'   Whether to convert to a pure abstract tensor, i.e., without any concrete data.
#' @return [`AbstractTensor`]
#' @export
to_abstract <- function(x, pure = FALSE) {
  x <- if (is_anvil_tensor(x)) {
    ConcreteTensor(x)
  } else if (is_abstract_tensor(x)) {
    x
  } else if (test_atomic(x) && (is.logical(x) || is.numeric(x))) {
    # logicals are not ambiguous
    LiteralTensor(x, integer(), ambiguous = !is.logical(x))
  } else if (is_graph_box(x)) {
    gnode <- x$gnode
    gnode$aval
  } else if (is_debug_box(x)) {
    x$aval
  } else {
    cli_abort("internal error: {.cls {class(x)}} is not a tensor-like object")
  }
  if (pure && class(x)[[1L]] != "AbstractTensor") {
    AbstractTensor(dtype = x$dtype, shape = x$shape, ambiguous = x$ambiguous)
  } else {
    x
  }
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
  inherits(x, "Shape")
}


#' @title Tensor-like Objects
#' @description
#' A value that is either an [`AnvilTensor`][nv_tensor], can be converted to it, or
#' represents an abstract version of it.
#' This also includes atomic R vectors.
#'
#' @name tensorish
#' @seealso [nv_tensor], [ConcreteTensor], [AbstractTensor], [LiteralTensor], [GraphBox]
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, dtype = "f32")
#' x
NULL

is_tensorish <- function(x) {
  inherits(x, "AnvilTensor") || inherits(x, "AbstractTensor") || inherits(x, "LiteralTensor") || inherits(x, "GraphBox") || inherits(x, "DebugBox")
}
