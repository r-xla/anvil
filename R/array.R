#' @title AnvilArray
#' @description
#' The main array object.
#' Its type is determined by a data type and a shape.
#'
#' Operations such as `nv_add(x, y)` can be called directly on `AnvilArray` objects (eager mode)
#' or inside [`jit()`]-compiled functions for better performance.
#'
#' To compare whether two abstract arrays are equal, use [`eq_type()`].
#'
#' @section Extractors:
#' The following generic functions can be used to extract information from an `AnvilArray`:
#' - [`dtype()`][tengen::dtype]: Get the data type of the array.
#' - [`shape()`][tengen::shape]: Get the shape (dimensions) of the array.
#' - [`ndims()`][tengen::ndims]: Get the number of dimensions.
#' - [`device()`][tengen::device]: Get the device of the array.
#' - [`platform()`]: Get the platform (e.g. `"cpu"`, `"cuda"`).
#' - [`ambiguous()`]: Get whether the dtype is ambiguous.
#'
#' @section Serialization:
#' Arrays can be serialized to and from the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format:
#' - [`nv_save()`] / [`nv_read()`]: Save/load arrays to/from a file.
#' - [`nv_serialize()`] / [`nv_unserialize()`]:
#'   Serialize/deserialize arrays to/from raw vectors.
#'
#' @seealso [nv_fill], [nv_iota], [nv_seq], [as_array], [nv_serialize]
#'
#' @param data (any)\cr
#'   `integer()`, `double()`, or `logical()` scalar, vector, or array.
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   One of `r stablehlo:::roxy_dtypes()` or a [`tengen::DataType`].
#'   The default (`NULL`) uses the current backend's default dtype:
#'   `f32` for numeric data on `"xla"`, `f64` for numeric data on `"quickr"`,
#'   `i32` for integer data, and `bool` for logical data.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device for the array (`"cpu"`, `"cuda"`).
#'   Default is to use the CPU for new arrays.
#'   This can be changed by setting the `PJRT_PLATFORM` environment variable.
#' @param shape (`NULL` | `integer()`)\cr
#'   The output shape of the array.
#'   The default (`NULL`) is to infer it from the data if possible.
#'   Note that [`nv_array`] interprets length 1 vectors as having shape `(1)`.
#'   To create a "scalar" with dimension `()`, use [`nv_scalar`] or explicitly specify `shape = c()`.
#' @param ambiguous (`NULL` | `logical(1)`)\cr
#'   Whether the dtype should be marked as ambiguous.
#'   Defaults to `FALSE` for new arrays.
#' @param backend (`NULL` | `character(1)`)\cr
#'   Backend to use (`"xla"` or `"quickr"`).
#'   Defaults to `default_backend()`.
#'   Must not be specified inside [`jit()`].
#' @return ([`AnvilArray`])
#' @examplesIf pjrt::plugins_downloaded()
#' # A 1-d array (vector) with shape (4). Default type for integers is `i32`
#' nv_array(1:4)
#'
#' # Specify a dtype
#' nv_array(c(1.5, 2.5, 3.5), dtype = "f64")
#'
#' # A 2x3 matrix
#' nv_array(1:6, shape = c(2L, 3L))
#'
#' # A scalar array.
#' nv_scalar(3.14)
#'
#' # A 0x3 array
#' nv_empty("f32", shape = c(0L, 3L))
#'
#' # --- Extractors ---
#' x <- nv_array(1:6, shape = c(2L, 3L))
#' dtype(x)
#' shape(x)
#' ndims(x)
#' device(x)
#' platform(x)
#' ambiguous(x)
#'
#' # --- Transforming arrays with jit ---
#' add_one <- jit(function(x) x + 1)
#' add_one(nv_array(1:4))
#'
#' # --- Eager mode (calling operations directly) ---
#' nv_add(nv_array(1:3), nv_array(4:6))
#'
#' @name AnvilArray
NULL

#' @rdname AnvilArray
#' @export
nv_array <- function(data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL, backend = NULL) {
  if (is_anvil_array(data)) {
    if (!is.null(device) && !eq_device(device(data), nv_device(device, backend))) {
      cli_abort("Cannot change device of existing AnvilArray from {.val {device(data)}} to {.val {device}}")
    }
    if (!is.null(shape) && !identical(shape(data), as.integer(shape))) {
      cli_abort("Cannot change shape of existing AnvilArray")
    }
    if (!is.null(dtype)) {
      if (dtype(data) != as_dtype(dtype)) {
        cli_abort("Cannot change dtype of existing AnvilArray from {.val {dtype(data)}} to {.val {dtype}}")
      }
    }
    if (!is.null(ambiguous) && ambiguous(data) != ambiguous) {
      cli_abort("Cannot change ambiguous of existing AnvilArray from {.val {ambiguous(data)}} to {.val {ambiguous}}")
    }
    return(data)
  }
  if (is.null(ambiguous)) {
    ambiguous <- FALSE
  }
  if (!is.null(dtype)) {
    dtype <- as_dtype(dtype)
  }
  if (!is.null(shape)) {
    shape <- as.integer(shape)
  }
  if (currently_tracing() && is.null(device)) {
    # The functions we jit should be backend-agnostic
    if (!is.null(backend)) {
      cli_abort("{.arg backend} must not be specified when calling {.fn nv_array} inside {.fn jit}.")
    }
    return(globals$backends[["plain"]]$new_data(data, dtype, shape, device, ambiguous))
  }
  if (is.null(backend) && is_device(device)) {
    backend <- backend(device)
  }
  backend <- backend %||% default_backend()
  globals$backends[[backend]]$new_data(data, dtype, shape, device, ambiguous)
}

#' @rdname AnvilArray
#' @param like ([`AnvilArray`])\cr
#'   An existing array. Any of `dtype`, `device`, `shape`, `ambiguous`, and
#'   `backend` that are `NULL` (the default) are taken from `like`.
#' @export
nv_array_like <- function(like, data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL, backend = NULL) {
  dtype <- dtype %||% dtype(like)
  device <- device %||% device(like)
  shape <- shape %||% shape(like)
  ambiguous <- ambiguous %||% ambiguous(like)
  backend <- backend %||% backend(like)
  nv_array(data = data, dtype = dtype, device = device, shape = shape, ambiguous = ambiguous, backend = backend)
}

is_anvil_array <- function(x) {
  inherits(x, "AnvilArray")
}

#' Get the underlying PJRT buffer from an AnvilArray or pass through other values
#' @param x An AnvilArray or any other value
#' @return The underlying PJRT buffer if x is an AnvilArray, otherwise x unchanged
#' @keywords internal
unwrap_if_array <- function(x) {
  if (is_anvil_array(x)) {
    x$data
  } else {
    x
  }
}

#' @rdname AnvilArray
#' @export
nv_scalar <- function(data, dtype = NULL, device = NULL, ambiguous = NULL, backend = NULL) {
  nv_array(data, dtype = dtype, device = device, shape = integer(), ambiguous = ambiguous, backend = backend)
}

#' @rdname AnvilArray
#' @export
nv_scalar_like <- function(like, data, dtype = NULL, device = NULL, ambiguous = NULL, backend = NULL) {
  dtype <- dtype %||% dtype(like)
  device <- device %||% device(like)
  ambiguous <- ambiguous %||% ambiguous(like)
  backend <- backend %||% backend(like)
  nv_scalar(data, dtype = dtype, device = device, ambiguous = ambiguous, backend = backend)
}

#' @rdname AnvilArray
#' @export
nv_empty <- function(dtype, shape, device = NULL, ambiguous = FALSE) {
  shape <- as.integer(shape)
  storage_mode <- switch(
    substr(as.character(if (is_dtype(dtype)) dtype else as_dtype(dtype)), 1L, 1L),
    "f" = "double",
    "i" = ,
    "u" = "integer",
    "b" = "logical",
    "double"
  )
  data <- array(vector(storage_mode, prod(shape)), dim = shape)
  nv_array(data, dtype = dtype, device = device, shape = shape, ambiguous = ambiguous)
}

#' @rdname AbstractArray
#' @export
nv_aval <- function(dtype, shape, ambiguous = FALSE) {
  AbstractArray(dtype = dtype, shape = shape, ambiguous = ambiguous)
}

#' @export
dtype.AnvilArray <- function(x, ...) {
  globals$backends[[x$backend]]$dtype(x)
}

#' @title Get Ambiguity of an Array
#' @description
#' Returns whether the array's dtype is ambiguous.
#' @param x An array object
#' @param ... Additional arguments (unused)
#' @return `logical(1)` - `TRUE` if the dtype is ambiguous, `FALSE` otherwise
#' @export
ambiguous <- function(x, ...) {
  UseMethod("ambiguous")
}

#' @export
ambiguous.AnvilArray <- function(x, ...) {
  globals$backends[[x$backend]]$ambiguous(x)
}

#' @export
ambiguous.AbstractArray <- function(x, ...) {
  x$ambiguous
}

#' @export
shape.AnvilArray <- function(x, ...) {
  globals$backends[[x$backend]]$shape(x)
}

#' @export
as_array.AnvilArray <- function(x, ...) {
  globals$backends[[x$backend]]$as_array(x)
}

#' @export
as_raw.AnvilArray <- function(x, row_major = FALSE, ...) {
  globals$backends[[x$backend]]$as_raw(x, row_major)
}

#' @rdname platform
#' @export
platform.AnvilArray <- function(x, ...) {
  globals$backends[[x$backend]]$platform(x)
}

#' @export
device.AnvilArray <- function(x, ...) {
  globals$backends[[x$backend]]$device(x)
}

#' @title Get Backend of an Array
#' @param x An array object
#' @param ... Additional arguments (unused)
#' @return `character(1)` - the backend name
#' @export
backend <- function(x, ...) {
  UseMethod("backend")
}

#' @export
backend.AnvilArray <- function(x, ...) {
  x$backend
}

#' @export
backend.PJRTDevice <- function(x, ...) {
  "xla"
}

#' @export
backend.QuickrDevice <- function(x, ...) {
  "quickr"
}

#' @title Abstract Array Class
#' @description
#' Representation of an abstract array type.
#' During tracing, it is wrapped in a [`GraphNode`] held by a [`GraphBox`].
#' In the lowered [`AnvilGraph`] it is also part of [`GraphNode`]s representing the values in the program.
#'
#' The base class represents an *unknown* value, but child classes exist for:
#' * closed-over constants: [`ConcreteArray`]
#' * scalar arrays arising from R literals: [`LiteralArray`]
#' * sequence patterns: [`IotaArray`]
#'
#' To convert a [`arrayish`] value to an abstract array, use [`to_abstract()`].
#'
#' @section Extractors:
#' The following extractors are available on `AbstractArray` objects:
#' - [`dtype()`][tengen::dtype]: Get the data type of the array.
#' - [`shape()`][tengen::shape]: Get the shape (dimensions) of the array.
#' - [`ambiguous()`]: Get whether the dtype is ambiguous.
#' - [`ndims()`][tengen::ndims]: Get the number of dimensions.
#'
#' @param dtype ([`tengen::DataType`] | `character(1)`)\cr
#'   The data type of the array.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the array. Can be provided as an integer vector.
#' @template param_ambiguous
#' @seealso [LiteralArray], [ConcreteArray], [IotaArray], [GraphValue], [to_abstract()], [GraphBox]
#'
#' @examplesIf pjrt::plugins_downloaded()
#' # -- Creating abstract arrays --
#' a <- AbstractArray("f32", c(2L, 3L))
#' a
#' dtype(a)
#' shape(a)
#' ambiguous(a)
#'
#' # Shorthand
#' nv_aval("f32", c(2L, 3L))
#'
#' # How AbstractArrays appear in an AnvilGraph
#' graph <- trace_fn(function(x) x + 1, list(x = nv_aval("i32", 4L)))
#' graph
#' graph$inputs[[1]]$aval
#'
#' @export
AbstractArray <- function(dtype, shape, ambiguous = FALSE) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  if (!test_flag(ambiguous)) {
    cli_abort("ambiguous must be a flag")
  }

  structure(
    list(dtype = dtype, shape = shape, ambiguous = ambiguous),
    class = "AbstractArray"
  )
}

is_abstract_tensor <- function(x) {
  inherits(x, "AbstractArray")
}

is_concrete_tensor <- function(x) {
  inherits(x, "ConcreteArray")
}

#' @method dtype AbstractArray
#' @export
dtype.AbstractArray <- function(x, ...) {
  x$dtype
}

#' @method shape AbstractArray
#' @export
shape.AbstractArray <- function(x, ...) {
  x$shape$dims
}

#' @title Concrete Array Class
#' @description
#' An [`AbstractArray`] that also holds a reference to the actual array data.
#' Usually represents a closed-over constant in a program.
#' Inherits from [`AbstractArray`].
#'
#' @section Lowering:
#' When lowering to XLA, these become inputs to the executable instead of embedding them into
#' programs as constants.
#' This is to avoid increasing compilation time and bloating the size of the executable.
#'
#' @param data ([`AnvilArray`])\cr
#'   The actual array data.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' y <- nv_array(c(0.5, 0.6))
#' x <- ConcreteArray(y)
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
ConcreteArray <- function(data) {
  if (!inherits(data, "AnvilArray")) {
    cli_abort("data must be an AnvilArray")
  }

  structure(
    list(
      dtype = dtype_from_buffer(data),
      shape = Shape(shape(data)),
      data = data,
      ambiguous = ambiguous(data)
    ),
    class = c("ConcreteArray", "AbstractArray")
  )
}

#' @title Literal Array Class
#' @description
#' An [`AbstractArray`] where all elements have the same constant value.
#' This either arises when using literals in traced code (e.g. `x + 1`) or when using
#' [`nv_fill()`] to create a constant.
#'
#' @section Type Ambiguity:
#' When arising from R literals, the resulting `LiteralArray` is ambiguous because no type
#' information was available. See the `vignette("type-promotion")` for more details.
#'
#' @section Lowering:
#' `LiteralArray`s become constants inlined into the stableHLO program.
#' I.e., they lower to [`stablehlo::hlo_tensor()`].
#'
#' @param data (`double(1)` | `integer(1)` | `logical(1)` | [`AnvilArray`])\cr
#'   The scalar value or scalarish AnvilArray (contains 1 element).
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the array.
#' @param dtype ([`tengen::DataType`])\cr
#'   The data type. Defaults to the current backend's default floating dtype,
#'   `i32` for integer, and `bool` for logical.
#' @template param_ambiguous
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- LiteralArray(1L, shape = integer(), ambiguous = TRUE)
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
LiteralArray <- function(data, shape, dtype = default_dtype(data), ambiguous) {
  if (!test_scalar(data) && !inherits(data, "AnvilArray")) {
    cli_abort("LiteralArrays expect scalars or AnvilArray")
  }
  if (inherits(data, "AnvilArray")) {
    if (prod(shape(data)) != 1L) {
      cli_abort("AnvilArray must contain exactly one element.")
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
    class = c("LiteralArray", "AbstractArray")
  )
}

#' @title Iota Array Class
#' @description
#' An [`AbstractArray`] representing an integer sequence.
#' Usually created by [`nv_iota()`] / [`nv_seq()`], which both call [`nvl_iota()`] internally.
#' Inherits from [`AbstractArray`].
#'
#' @section Lowering:
#' When lowering to stableHLO, these become `iota` operations that generate the integer sequence
#' so they do not need to actually hold the data in the executable, similar to `ALTREP`s in R.
#' It lowers to [`stablehlo::hlo_iota()`], optionally shifting the starting value via
#' [`stablehlo::hlo_add()`].
#'
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the array.
#' @param dtype ([`tengen::DataType`])\cr
#'   The data type.
#' @param dimension (`integer(1)`)\cr
#'   The dimension along which values increase.
#' @param start (`integer(1)`)\cr
#'   The starting value.
#' @template param_ambiguous
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- IotaArray(shape = 4L, dtype = "i32", dimension = 1L)
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
IotaArray <- function(shape, dtype, dimension, start = 1L, ambiguous = FALSE) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  assert_flag(ambiguous)
  # stablehlo::Shape is a wrapper object; its rank is length(shape$dims), not length(shape)
  assert_int(dimension, lower = 1L, upper = length(shape$dims))
  assert_int(start)
  structure(
    list(shape = shape, dtype = dtype, dimension = dimension, start = start, ambiguous = ambiguous),
    class = c("IotaArray", "AbstractArray")
  )
}

#' @export
format.IotaArray <- function(x, ...) {
  sprintf(
    "IotaArray(shape=%s, dtype=%s, dimension=%s, start=%s)",
    shape2string(x$shape),
    dtype2string(x$dtype, x$ambiguous),
    x$dimension,
    x$start
  )
}

#' @export
print.IotaArray <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
`==.AbstractArray` <- function(e1, e2) {
  cli_abort("Use {.fn eq_type} instead of {.code ==} for comparing AbstractArrays")
}

#' @export
`!=.AbstractArray` <- function(e1, e2) {
  cli_abort("Use {.fn neq_type} instead of {.code !=} for comparing AbstractArrays")
}

#' @title Compare AbstractArray Types
#' @description
#' Compare two abstract arrays for type equality.
#' @param e1 ([`AbstractArray`])\cr
#'   First array to compare.
#' @param e2 ([`AbstractArray`])\cr
#'   Second array to compare.
#' @param ambiguity (`logical(1)`)\cr
#'   Whether to consider the ambiguous field when comparing.
#'   If `TRUE`, arrays with different ambiguity are not equal.
#'   If `FALSE`, only dtype and shape are compared.
#' @return `logical(1)` - `TRUE` if the arrays are equal, `FALSE` otherwise.
#' @examples
#' a <- nv_aval("f32", c(2L, 3L))
#' b <- nv_aval("f32", c(2L, 3L))
#'
#' # Same dtype and shape
#' eq_type(a, b, ambiguity = FALSE)
#'
#' # Different dtype
#' eq_type(a, nv_aval("i32", c(2L, 3L)), ambiguity = FALSE)
#'
#' # Different shape
#' eq_type(a, nv_aval("f32", c(3L, 2L)), ambiguity = FALSE)
#'
#' # ambiguity parameter controls whether ambiguous field is compared
#' c <- nv_aval("f32", c(2L, 3L), ambiguous = TRUE)
#' eq_type(a, c, ambiguity = FALSE)
#' eq_type(a, c, ambiguity = TRUE)
#'
#' # neq_type is the negation of eq_type
#' neq_type(a, b, ambiguity = FALSE)
#' @export
eq_type <- function(e1, e2, ambiguity) {
  if (!inherits(e1, "AbstractArray") || !inherits(e2, "AbstractArray")) {
    cli_abort("e1 and e2 must be AbstractArrays")
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
repr.AbstractArray <- function(x, ...) {
  sprintf("%s[%s]", paste0(repr(x$dtype), if (x$ambiguous) "?"), repr(x$shape))
}

#' @export
format.AbstractArray <- function(x, ...) {
  sprintf(
    "AbstractArray(dtype=%s, shape=%s)",
    if (x$ambiguous) paste0(repr(x$dtype), "?") else repr(x$dtype),
    repr(x$shape)
  )
}

#' @export
format.ConcreteArray <- function(x, ...) {
  sprintf("ConcreteArray(%s, %s)", dtype2string(x$dtype, x$ambiguous), shape2string(x$shape))
}

#' @export
format.LiteralArray <- function(x, ...) {
  data_str <- if (is_anvil_array(x$data)) {
    trimws(capture.output(print(x$data, ..., header = FALSE))[1L])
  } else {
    x$data
  }
  sprintf("LiteralArray(%s, %s, %s)", data_str, dtype2string(x$dtype, x$ambiguous), shape2string(x$shape))
}

#' @export
print.AbstractArray <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
print.ConcreteArray <- function(x, ...) {
  cat("ConcreteArray\n")
  print(x$data, header = FALSE)
  invisible(x)
}

#' @export
format.AnvilArray <- function(x, ...) {
  dtype_str <- if (ambiguous(x)) paste0(repr(dtype(x)), "?") else repr(dtype(x))
  sprintf("AnvilArray(dtype=%s, shape=%s)", dtype_str, paste(shape(x), collapse = "x"))
}

#' @export
print.AnvilArray <- function(x, header = TRUE, ...) {
  if (header) {
    cat("AnvilArray\n")
  }
  dtype_str <- paste0(as.character(dtype(x)), if (ambiguous(x)) "?")
  footer <- sprintf("[ %s%s{%s} ]", toupper(platform(x)), dtype_str, paste0(shape(x), collapse = ","))
  globals$backends[[x$backend]]$print_data(x, footer)
  invisible(x)
}

# fmt: skip
compare_proxy.AnvilArray <- function(x, path) { # nolint
  list(
    object = list(
      data = as_array(x),
      dtype = as.character(dtype(x)),
      ambiguous = ambiguous(x),
      backend = backend(x)
    ),
    path = path
  )
}

#' @title Convert to Abstract Array
#' @description
#' Convert an object to its abstract array representation ([`AbstractArray`]).
#' @param x (`any`)\cr
#'   Object to convert.
#' @param pure (`logical(1)`)\cr
#'   Whether to convert to a pure `AbstractArray` and not e.g. `LiteralArray` or `ConcreteArray`.
#' @return [`AbstractArray`]
#' @examplesIf pjrt::plugins_downloaded()
#' # R literals become LiteralArrays (ambiguous by default, except logicals)
#' to_abstract(1.5)
#' to_abstract(1L)
#' to_abstract(TRUE)
#'
#' # AnvilArrays become ConcreteArrays
#' to_abstract(nv_array(1:4))
#'
#' # Use pure = TRUE to strip subclass info
#' to_abstract(nv_array(1:4), pure = TRUE)
#'
#' @export
to_abstract <- function(x, pure = FALSE) {
  x <- if (is_anvil_array(x)) {
    ConcreteArray(x)
  } else if (is_abstract_tensor(x)) {
    x
  } else if (test_atomic(x) && (is.logical(x) || is.numeric(x))) {
    # logicals are not ambiguous
    LiteralArray(x, integer(), ambiguous = !is.logical(x))
  } else if (is_graph_box(x)) {
    gnode <- x$gnode
    gnode$aval
  } else {
    cli_abort("internal error: {.cls {class(x)}} is not an array-like object")
  }
  if (pure && class(x)[[1L]] != "AbstractArray") {
    AbstractArray(dtype = x$dtype, shape = x$shape, ambiguous = x$ambiguous)
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


#' @title Array-like Objects
#' @description
#' A `arrayish` value is any object that can be input to a primitive such as [`nvl_add`].
#'
#' During runtime of a JIT-compiled function, these are [`AnvilArray`] objects.
#'
#' The following types are arrayish (during tracing):
#' * [`AnvilArray`]: a concrete array holding data on a device.
#' * [`GraphBox`]: a boxed abstract array representing a value in a graph.
#' * Length-1 vectors: `numeric(1)` and `logical(1)`
#' * R arrays of types: `numeric` and `logical`.
#'
#' Use [`is_arrayish()`] to check whether a value is arrayish.
#'
#' @param x (`any`)\cr
#'   Object to check.
#' @param convert_ok (`logical(1)`)\cr
#'   Whether to accept `numeric(1)` and `logical(1)` and R arrays of type `numeric` and `logical`.
#' @return `logical(1)`
#' @name arrayish
#' @seealso [AnvilArray], [GraphBox]
#' @examplesIf pjrt::plugins_downloaded()
#' # AnvilArrays are arrayish
#' is_arrayish(nv_array(1:4))
#'
#' # Scalar R literals are arrayish by default
#' is_arrayish(1.5)
#' # R arrays are arrayish by default
#' is_arrayish(array(1.5))
#'
#' # R arrays
#' is_arrayish(array(1:4), convert_ok = TRUE)
#' is_arrayish(array(1:4), convert_ok = FALSE)
#'
#' # Length 1 vectors
#' is_arrayish(1.5, convert_ok = FALSE)
#' is_arrayish(1.5, convert_ok = TRUE)
NULL

#' @rdname arrayish
#' @export
is_arrayish <- function(x, convert_ok = TRUE) {
  ok <- inherits(x, "AnvilArray") ||
    is_box(x)

  if (ok) {
    return(TRUE)
  }

  if (!convert_ok) {
    return(FALSE)
  }
  is_valid_r(x)
}
