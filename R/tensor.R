#' @title AnvilTensor
#' @description
#' The main tensor object.
#' Its type is determined by a data type and a shape.
#'
#' To transform tensors, apply [`jit()`]ted functions.
#' Directly calling operations (e.g. `nv_add(x, y)`) on `AnvilTensor` objects
#' only performs type inference and returns an [`AbstractTensor`] --
#' see `vignette("debugging")` for details.
#'
#' To compare whether two abstract tensors are equal, use [`eq_type()`].
#'
#' @section Extractors:
#' The following generic functions can be used to extract information from an `AnvilTensor`:
#' - [`dtype()`][tengen::dtype]: Get the data type of the tensor.
#' - [`shape()`][tengen::shape]: Get the shape (dimensions) of the tensor.
#' - [`ndims()`][tengen::ndims]: Get the number of dimensions.
#' - [`device()`][tengen::device]: Get the device of the tensor.
#' - [`platform()`][pjrt::platform]: Get the platform (e.g. `"cpu"`, `"cuda"`).
#' - [`ambiguous()`]: Get whether the dtype is ambiguous.
#'
#' @section Serialization:
#' Tensors can be serialized to and from the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format:
#' - [`nv_save()`] / [`nv_read()`]: Save/load tensors to/from a file.
#' - [`nv_serialize()`] / [`nv_unserialize()`]:
#'   Serialize/deserialize tensors to/from raw vectors.
#'
#' @seealso [nv_fill], [nv_iota], [nv_seq], [as_array], [nv_serialize]
#'
#' @param data (any)\cr
#'   Object convertible to a [`PJRTBuffer`][pjrt::pjrt_buffer].
#'   Includes `integer()`, `double()`, `logical()` vectors and arrays.
#' @param dtype (`NULL` | `character(1)` | [`TensorDataType`])\cr
#'   One of `r stablehlo:::roxy_dtypes()` or a [`stablehlo::TensorDataType`].
#'   The default (`NULL`) uses `f32` for numeric data, `i32` for integer data, and `i1` for logical data.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device for the tensor (`"cpu"`, `"cuda"`).
#'   Default is to use the CPU for new tensors.
#'   This can be changed by setting the `PJRT_PLATFORM` environment variable.
#' @param shape (`NULL` | `integer()`)\cr
#'   The output shape of the tensor.
#'   The default (`NULL`) is to infer it from the data if possible.
#'   Note that [`nv_tensor`] interprets length 1 vectors as having shape `(1)`.
#'   To create a "scalar" with dimension `()`, use [`nv_scalar`] or explicitly specify `shape = c()`.
#' @param ambiguous (`NULL` | `logical(1)`)\cr
#'   Whether the dtype should be marked as ambiguous.
#'   Defaults to `FALSE` for new tensors.
#' @return ([`AnvilTensor`])
#' @examplesIf pjrt::plugin_is_downloaded()
#' # A 1-d tensor (vector) with shape (4). Default type for integers is `i32`
#' nv_tensor(1:4)
#'
#' # Specify a dtype
#' nv_tensor(c(1.5, 2.5, 3.5), dtype = "f64")
#'
#' # A 2x3 matrix
#' nv_tensor(1:6, shape = c(2L, 3L))
#'
#' # A scalar tensor.
#' nv_scalar(3.14)
#'
#' # A 0x3 tensor
#' nv_empty("f32", shape = c(0L, 3L))
#'
#' # --- Extractors ---
#' x <- nv_tensor(1:6, shape = c(2L, 3L))
#' dtype(x)
#' shape(x)
#' ndims(x)
#' device(x)
#' platform(x)
#' ambiguous(x)
#'
#' # --- Transforming tensors with jit ---
#' add_one <- jit(function(x) x + 1)
#' add_one(nv_tensor(1:4))
#'
#' # --- Debug mode (calling operations directly) ---
#' # Outside of jit, operations only perform type inference:
#' nv_add(nv_tensor(1:3), nv_tensor(4:6))
#'
#' @name AnvilTensor
NULL

#' @rdname AnvilTensor
#' @export
nv_tensor <- function(data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL) {
  if (is_anvil_tensor(data)) {
    if (!is.null(device) && device(data) != pjrt::as_pjrt_device(device)) {
      cli_abort("Cannot change device of existing AnvilTensor from {.val {platform(data)}} to {.val {device}}")
    }
    if (!is.null(shape) && !identical(shape(data), as.integer(shape))) {
      cli_abort("Cannot change shape of existing AnvilTensor")
    }
    if (!is.null(dtype)) {
      if (dtype(data) != as_dtype(dtype)) {
        cli_abort("Cannot change dtype of existing AnvilTensor from {.val {dtype(data)}} to {.val {dtype}}")
      }
    }
    if (!is.null(ambiguous) && ambiguous(data) != ambiguous) {
      cli_abort("Cannot change ambiguous of existing AnvilTensor from {.val {ambiguous(data)}} to {.val {ambiguous}}")
    }
    return(data)
  }
  if (is.null(ambiguous)) {
    ambiguous <- FALSE
  }
  if (is_dtype(dtype)) {
    dtype <- as.character(dtype)
  }
  x <- pjrt_buffer(data, dtype, device = device, shape = shape)
  ensure_nv_tensor(x, ambiguous = ambiguous)
}

is_anvil_tensor <- function(x) {
  inherits(x, "AnvilTensor")
}

#' Get the underlying PJRT buffer from an AnvilTensor or pass through other values
#' @param x An AnvilTensor or any other value
#' @return The underlying PJRT buffer if x is an AnvilTensor, otherwise x unchanged
#' @keywords internal
unwrap_if_tensor <- function(x) {
  if (is_anvil_tensor(x)) {
    x$tensor
  } else {
    x
  }
}

ensure_nv_tensor <- function(x, ambiguous = FALSE) {
  if (inherits(x, "AnvilTensor")) {
    if (ambiguous != x$ambiguous) {
      x$ambiguous <- ambiguous
    }
    return(x)
  }
  assert_class(x, "PJRTBuffer")
  structure(
    list(tensor = x, ambiguous = ambiguous),
    class = "AnvilTensor"
  )
}

#' @rdname AnvilTensor
#' @export
nv_scalar <- function(data, dtype = NULL, device = NULL, ambiguous = NULL) {
  if (is.null(ambiguous)) {
    ambiguous <- FALSE
  }
  if (is_dtype(dtype)) {
    dtype <- as.character(dtype)
  }
  x <- pjrt_scalar(data, dtype = dtype, device = device)
  ensure_nv_tensor(x, ambiguous = ambiguous)
}

#' @rdname AnvilTensor
#' @export
nv_empty <- function(dtype, shape, device = NULL, ambiguous = FALSE) {
  if (is_dtype(dtype)) {
    dtype <- as.character(dtype)
  }
  x <- pjrt::pjrt_empty(dtype, shape, device = device)
  ensure_nv_tensor(x, ambiguous = ambiguous)
}

#' @rdname AbstractTensor
#' @export
nv_aten <- function(dtype, shape, ambiguous = FALSE) {
  AbstractTensor(dtype = dtype, shape = shape, ambiguous = ambiguous)
}

#' @export
dtype.AnvilTensor <- function(x, ...) {
  as_dtype(as.character(pjrt::elt_type(x$tensor)))
}

#' @title Get Ambiguity of a Tensor
#' @description
#' Returns whether the tensor's dtype is ambiguous.
#' @param x A tensor object
#' @param ... Additional arguments (unused)
#' @return `logical(1)` - `TRUE` if the dtype is ambiguous, `FALSE` otherwise
#' @export
ambiguous <- function(x, ...) {
  UseMethod("ambiguous")
}

#' @export
ambiguous.AnvilTensor <- function(x, ...) {
  x$ambiguous
}

#' @export
ambiguous.AbstractTensor <- function(x, ...) {
  x$ambiguous
}

#' @export
shape.AnvilTensor <- function(x, ...) {
  tengen::shape(x$tensor)
}

#' @export
as_array.AnvilTensor <- function(x, ...) {
  tengen::as_array(x$tensor)
}

#' @export
as_raw.AnvilTensor <- function(x, row_major = FALSE, ...) {
  tengen::as_raw(x$tensor, row_major = row_major)
}

#' @method ndims AnvilTensor
#' @export
ndims.AnvilTensor <- function(x, ...) {
  tengen::ndims(x$tensor)
}

#' @export
platform.AnvilTensor <- function(x, ...) {
  pjrt::platform(x$tensor)
}

#' @export
device.AnvilTensor <- function(x, ...) {
  device(x$tensor)
}

#' @title Abstract Tensor Class
#' @description
#' Representation of an abstract tensor type.
#' During tracing, it is wrapped in a [`GraphNode`] held by a [`GraphBox`].
#' In the lowered [`AnvilGraph`] it is also part of [`GraphNode`]s representing the values in the program.
#'
#' The base class represents an *unknown* value, but child classes exist for:
#' * closed-over constants: [`ConcreteTensor`]
#' * scalar tensors arising from R literals: [`LiteralTensor`]
#' * sequence patterns: [`IotaTensor`]
#'
#' To convert a [`tensorish`] value to an abstract tensor, use [`to_abstract()`].
#'
#' @section Extractors:
#' The following extractors are available on `AbstractTensor` objects:
#' - [`dtype()`][tengen::dtype]: Get the data type of the tensor.
#' - [`shape()`][tengen::shape]: Get the shape (dimensions) of the tensor.
#' - [`ambiguous()`]: Get whether the dtype is ambiguous.
#' - [`ndims()`][tengen::ndims]: Get the number of dimensions.
#'
#' @param dtype ([`stablehlo::TensorDataType`] | `character(1)`)\cr
#'   The data type of the tensor.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the tensor. Can be provided as an integer vector.
#' @template param_ambiguous
#' @seealso [LiteralTensor], [ConcreteTensor], [IotaTensor], [GraphValue], [to_abstract()], [GraphBox]
#'
#' @examplesIf pjrt::plugin_is_downloaded()
#' # -- Creating abstract tensors --
#' a <- AbstractTensor("f32", c(2L, 3L))
#' a
#' dtype(a)
#' shape(a)
#' ambiguous(a)
#'
#' # Shorthand
#' nv_aten("f32", c(2L, 3L))
#'
#' # How AbstractTensors appear in an AnvilGraph
#' graph <- trace_fn(function(x) x + 1, list(x = nv_aten("i32", 4L)))
#' graph
#' graph$inputs[[1]]$aval
#'
#' @export
AbstractTensor <- function(dtype, shape, ambiguous = FALSE) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  if (!test_flag(ambiguous)) {
    cli_abort("ambiguous must be a flag")
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
#' @method platform AbstractTensor
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

#' @method ndims AbstractTensor
#' @export
ndims.AbstractTensor <- function(x, ...) {
  length(x$shape$dims)
}

#' @title Concrete Tensor Class
#' @description
#' An [`AbstractTensor`] that also holds a reference to the actual tensor data.
#' Usually represents a closed-over constant in a program.
#' Inherits from [`AbstractTensor`].
#'
#' @section Lowering:
#' When lowering to XLA, these become inputs to the executable instead of embedding them into
#' programs as constants.
#' This is to avoid increasing compilation time and bloating the size of the executable.
#'
#' @param data ([`AnvilTensor`])\cr
#'   The actual tensor data.
#'
#' @examplesIf pjrt::plugin_is_downloaded()
#' y <- nv_tensor(c(0.5, 0.6))
#' x <- ConcreteTensor(y)
#' x
#' ambiguous(x)
#' shape(x)
#' ndims(x)
#' dtype(x)
#'
#' # How it appears during tracing
#' graph <- trace_fn(function() y, list())
#' graph
#' graph$outputs[[1]]$aval
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
      ambiguous = ambiguous(data)
    ),
    class = c("ConcreteTensor", "AbstractTensor")
  )
}

#' @title Literal Tensor Class
#' @description
#' An [`AbstractTensor`] where all elements have the same constant value.
#' This either arises when using literals in traced code (e.g. `x + 1`) or when using
#' [`nv_fill()`] to create a constant.
#'
#' @section Type Ambiguity:
#' When arising from R literals, the resulting `LiteralTensor` is ambiguous because no type
#' information was available. See the `vignette("type-promotion")` for more details.
#'
#' @section Lowering:
#' `LiteralTensor`s become constants inlined into the stableHLO program.
#' I.e., they lower to [`stablehlo::hlo_tensor()`].
#'
#' @param data (`double(1)` | `integer(1)` | `logical(1)` | [`AnvilTensor`])\cr
#'   The scalar value or scalarish AnvilTensor (contains 1 element).
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the tensor.
#' @param dtype ([`stablehlo::TensorDataType`])\cr
#'   The data type. Defaults to `f32` for numeric, `i32` for integer, `i1` for logical.
#' @template param_ambiguous
#'
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- LiteralTensor(1L, shape = integer(), ambiguous = TRUE)
#' x
#' ambiguous(x)
#' shape(x)
#' ndims(x)
#' dtype(x)
#' # How it appears during tracing:
#' # 1. via R literals
#' graph <- trace_fn(function() 1, list())
#' graph
#' graph$outputs[[1]]$aval
#' # 2. via nv_fill()
#' graph <- trace_fn(function() nv_fill(2L, shape = c(2, 2)), list())
#' graph
#' graph$outputs[[1]]$aval
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

#' @title Iota Tensor Class
#' @description
#' An [`AbstractTensor`] representing an integer sequence.
#' Usually created by [`nv_iota()`] / [`nv_seq()`], which both call [`nvl_iota()`] internally.
#' Inherits from [`AbstractTensor`].
#'
#' @section Lowering:
#' When lowering to stableHLO, these become `iota` operations that generate the integer sequence
#' so they do not need to actually hold the data in the executable, similar to `ALTREP`s in R.
#' It lowers to [`stablehlo::hlo_iota()`], optionally shifting the starting value via
#' [`stablehlo::hlo_add()`].
#'
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the tensor.
#' @param dtype ([`stablehlo::TensorDataType`])\cr
#'   The data type.
#' @param dimension (`integer(1)`)\cr
#'   The dimension along which values increase.
#' @param start (`integer(1)`)\cr
#'   The starting value.
#' @template param_ambiguous
#'
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- IotaTensor(shape = 4L, dtype = "i32", dimension = 1L)
#' x
#' ambiguous(x)
#' shape(x)
#' ndims(x)
#' dtype(x)
#' # How it appears during tracing:
#' graph <- trace_fn(function() nv_iota(dim = 1L, dtype = "i32", shape = 4L), list())
#' graph
#' graph$outputs[[1]]$aval
#' @export
IotaTensor <- function(shape, dtype, dimension, start = 1L, ambiguous = FALSE) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  assert_flag(ambiguous)
  # stablehlo::Shape is a wrapper object; its rank is length(shape$dims), not length(shape)
  assert_int(dimension, lower = 1L, upper = length(shape$dims))
  assert_int(start)
  structure(
    list(shape = shape, dtype = dtype, dimension = dimension, start = start, ambiguous = ambiguous),
    class = c("IotaTensor", "AbstractTensor")
  )
}

#' @export
format.IotaTensor <- function(x, ...) {
  sprintf(
    "IotaTensor(shape=%s, dtype=%s, dimension=%s, start=%s)",
    shape2string(x$shape),
    dtype2string(x$dtype, x$ambiguous),
    x$dimension,
    x$start
  )
}

#' @export
print.IotaTensor <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

is_literal_tensor <- function(x) {
  inherits(x, "LiteralTensor")
}

#' @exportS3Method platform ConcreteTensor
platform.ConcreteTensor <- function(x, ...) {
  platform(x$data)
}

#' @export
`==.AbstractTensor` <- function(e1, e2) {
  cli_abort("Use {.fn eq_type} instead of {.code ==} for comparing AbstractTensors")
}

#' @export
`!=.AbstractTensor` <- function(e1, e2) {
  cli_abort("Use {.fn neq_type} instead of {.code !=} for comparing AbstractTensors")
}

#' @title Compare AbstractTensor Types
#' @description
#' Compare two AbstractTensors for type equality.
#' @param e1 ([`AbstractTensor`])\cr
#'   First tensor to compare.
#' @param e2 ([`AbstractTensor`])\cr
#'   Second tensor to compare.
#' @param ambiguity (`logical(1)`)\cr
#'   Whether to consider the ambiguous field when comparing.
#'   If `TRUE`, tensors with different ambiguity are not equal.
#'   If `FALSE`, only dtype and shape are compared.
#' @return `logical(1)` - `TRUE` if the tensors are equal, `FALSE` otherwise.
#' @examples
#' a <- nv_aten("f32", c(2L, 3L))
#' b <- nv_aten("f32", c(2L, 3L))
#'
#' # Same dtype and shape
#' eq_type(a, b, ambiguity = FALSE)
#'
#' # Different dtype
#' eq_type(a, nv_aten("i32", c(2L, 3L)), ambiguity = FALSE)
#'
#' # Different shape
#' eq_type(a, nv_aten("f32", c(3L, 2L)), ambiguity = FALSE)
#'
#' # ambiguity parameter controls whether ambiguous field is compared
#' c <- nv_aten("f32", c(2L, 3L), ambiguous = TRUE)
#' eq_type(a, c, ambiguity = FALSE)
#' eq_type(a, c, ambiguity = TRUE)
#'
#' # neq_type is the negation of eq_type
#' neq_type(a, b, ambiguity = FALSE)
#' @export
eq_type <- function(e1, e2, ambiguity) {
  if (!inherits(e1, "AbstractTensor") || !inherits(e2, "AbstractTensor")) {
    cli_abort("e1 and e2 must be AbstractTensors")
  }
  if (e1$dtype != e2$dtype || !identical(e1$shape, e2$shape)) {
    return(FALSE)
  }
  if (ambiguity && (e1$ambiguous != e2$ambiguous)) {
    return(FALSE)
  }
  TRUE
}

#' @rdname eq_type
#' @export
neq_type <- function(e1, e2, ambiguity) {
  !eq_type(e1, e2, ambiguity)
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
format.ConcreteTensor <- function(x, ...) {
  sprintf("ConcreteTensor(%s, %s)", dtype2string(x$dtype, x$ambiguous), shape2string(x$shape))
}

#' @export
format.LiteralTensor <- function(x, ...) {
  data_str <- if (is_anvil_tensor(x$data)) {
    trimws(capture.output(print(x$data, ..., header = FALSE))[1L])
  } else {
    x$data
  }
  sprintf("LiteralTensor(%s, %s, %s)", data_str, dtype2string(x$dtype, x$ambiguous), shape2string(x$shape))
}

#' @export
print.AbstractTensor <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
print.ConcreteTensor <- function(x, ...) {
  cat("ConcreteTensor\n")
  print(x$data, header = FALSE)
  invisible(x)
}

#' @export
format.AnvilTensor <- function(x, ...) {
  dtype_str <- if (x$ambiguous) paste0(repr(dtype(x)), "?") else repr(dtype(x))
  sprintf("AnvilTensor(dtype=%s, shape=%s)", dtype_str, paste(shape(x), collapse = "x"))
}

#' @export
print.AnvilTensor <- function(x, header = TRUE, ...) {
  if (header) {
    cat("AnvilTensor\n")
  }
  dtype_str <- paste0(as.character(dtype(x)), if (x$ambiguous) "?")
  footer <- sprintf("[ %s%s{%s} ]", toupper(platform(x)), dtype_str, paste0(shape(x), collapse = ","))

  print(x$tensor, header = FALSE, footer = footer)
  invisible(x)
}

# fmt: skip
compare_proxy.AnvilTensor <- function(x, path) { # nolint
  list(
    object = list(
      data = as_array(x),
      dtype = as.character(dtype(x)),
      ambiguous = ambiguous(x)
    ),
    path = path
  )
}

#' @title Convert to Abstract Tensor
#' @description
#' Convert an object to its abstract tensor representation ([`AbstractTensor`]).
#' @param x (`any`)\cr
#'   Object to convert.
#' @param pure (`logical(1)`)\cr
#'   Whether to convert to a pure `AbstractTensor` and not e.g. `LiteralTensor` or `ConcreteTensor`.
#' @return [`AbstractTensor`]
#' @examplesIf pjrt::plugin_is_downloaded()
#' # R literals become LiteralTensors (ambiguous by default, except logicals)
#' to_abstract(1.5)
#' to_abstract(1L)
#' to_abstract(TRUE)
#'
#' # AnvilTensors become ConcreteTensors
#' to_abstract(nv_tensor(1:4))
#'
#' # Use pure = TRUE to strip subclass info
#' to_abstract(nv_tensor(1:4), pure = TRUE)
#'
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
#' A `tensorish` value is any object that can be passed as an input to
#' anvil primitive functions such as [`nvl_add`] or is an output of such a function.
#'
#' During runtime, these are [`AnvilTensor`] objects.
#'
#' The following types are tensorish (during compile-time):
#' * [`AnvilTensor`]: a concrete tensor holding data on a device.
#' * [`GraphBox`]: a boxed abstract tensor representing a value in a graph.
#' * Literals: `numeric(1)`, `integer(1)`, `logical(1)`: promoted to scalar tensors.
#'
#' Use [`is_tensorish()`] to check whether a value is tensorish.
#'
#' @param x (`any`)\cr
#'   Object to check.
#' @param literal (`logical(1)`)\cr
#'   Whether to accept R literals as tensorish.
#' @return `logical(1)`
#' @name tensorish
#' @seealso [AnvilTensor], [GraphBox]
#' @examplesIf pjrt::plugin_is_downloaded()
#' # AnvilTensors are tensorish
#' is_tensorish(nv_tensor(1:4))
#'
#' # Scalar R literals are tensorish by default
#' is_tensorish(1.5)
#'
#' # Non-scalar vectors are not tensorish
#' is_tensorish(1:4)
#'
#' is_tensorish(DebugBox(nv_aten("f32", c(2L, 3L))))
#'
#' # Disable literal promotion
#' is_tensorish(1.5, literal = FALSE)
NULL

#' @rdname tensorish
#' @export
is_tensorish <- function(x, literal = TRUE) {
  ok <- inherits(x, "AnvilTensor") ||
    is_box(x)

  if (!ok && literal) {
    ok <- test_scalar(x) && (is.numeric(x) || is.logical(x))
  }
  return(ok)
}
