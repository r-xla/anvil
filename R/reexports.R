#' @title Get the shape of a tensor
#'
#' @description Returns the shape of a tensor as an `integer()` vector.
#'
#' @details This is implemented via the generic [`tengen::shape()`].
#'
#' @param x ([`tensorish`])\cr
#'   A tensor-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns `integer()`
#' @name shape
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, dtype = "f32")
#' shape(x)
NULL

#' @rdname shape
#' @importFrom tengen shape
#' @export
tengen::shape

#' @title Get the device of a tensor
#'
#' @description Returns the device on which a tensor is allocated.
#'
#' @details This is implemented via the generic [`tengen::device()`].
#'
#' @param x ([`tensorish`])\cr
#'   A tensor-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns [`PJRTDevice`][pjrt::pjrt_device]
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, dtype = "f32")
#' device(x)
#' @name device
NULL

#' @rdname device
#' @importFrom tengen device
#' @export
tengen::device

#' @title Convert a tensor to an R array
#'
#' @description
#' Transfers tensor data to R and returns it as an R [`array`].
#' Only in the case of scalars is the result a vector of length 1, as R `arrays` cannot have 0 dimensions.
#'
#' @details
#' This is implemented via the generic [`tengen::as_array()`].
#'
#' @param x ([`tensorish`])\cr
#'   A tensor-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns An R [`array`] or `vector` of length 1.
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, dtype = "f32")
#' as_array(x)
#' y <- nv_scalar(1L)
#' # R arrays can't have 0 dimensions:
#' as_array(y)
#' @name as_array
NULL

#' @rdname as_array
#' @importFrom tengen as_array
#' @export
tengen::as_array

#' @title Convert a tensor to a raw vector
#'
#' @description Returns the underlying bytes of a tensor as a [raw] vector.
#'
#' @details This is implemented via the generic [`tengen::as_raw()`].
#'
#' @param x ([`tensorish`])\cr
#'   A tensor-like object.
#' @param ... Additional arguments passed to method:
#'   - `row_major` (`logical(1)`)\cr
#'     Whether to write the bytes in row-major order.
#' @returns A [`raw`] vector.
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, shape = c(2, 2), dtype = "f32")
#' as_raw(x, row_major = TRUE)
#' as_raw(x, row_major = FALSE)
#' @name as_raw
NULL

#' @rdname as_raw
#' @importFrom tengen as_raw
#' @export
tengen::as_raw

#' @title Get the data type of a tensor
#'
#' @description
#' Returns the data type of a tensor (e.g. `f32`, `i64`).
#'
#' @details This is implemented via the generic [`tengen::dtype()`].
#'
#' @param x ([`tensorish`])\cr
#'   A tensor-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns A [`TensorDataType`][stablehlo::TensorDataType].
#' @seealso [tengen::dtype()]
#' @name dtype
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, dtype = "f32")
#' dtype(x)
NULL

#' @rdname dtype
#' @importFrom tengen dtype
#' @export
tengen::dtype

#' @title Get the number of dimensions of a tensor
#'
#' @description Returns the number of dimensions (sometimes also refered to as rank) of a tensor.
#' Equivalent to `length(shape(x))`.
#'
#' @param x ([`tensorish`])\cr
#'   A tensor-like object.
#' @returns `integer(1)`
#' @seealso [tengen::ndims()]
#' @name ndims
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, dtype = "f32")
#' ndims(x)
NULL

#' @rdname ndims
#' @importFrom tengen ndims
#' @export
tengen::ndims

#' @title Check if an object is a TensorDataType
#'
#' @description Tests whether `x` is a `TensorDataType` object.
#'
#' @param x An object to test.
#' @returns `TRUE` or `FALSE`.
#' @seealso [as_dtype()], [stablehlo::is_dtype()]
#' @name is_dtype
#' @examples
#' is_dtype("f32")
#' is_dtype(as_dtype("f32"))
NULL

#' @rdname is_dtype
#' @importFrom stablehlo is_dtype
#' @export
stablehlo::is_dtype

#' @title Convert to a TensorDataType
#'
#' @description Coerces a value to a `TensorDataType`. Accepts data type strings
#' (e.g. `"f32"`, `"i64"`, `"i1"`) or existing `TensorDataType` objects (they are returned unchanged).
#'
#' @details
#' This is implemented via the generic [`stablehlo::as_dtype()`].
#'
#' @param x A character string or `TensorDataType` to convert.
#' @returns A `TensorDataType` object.
#' @seealso [is_dtype()], [stablehlo::as_dtype()], [`stablehlo::TensorDataType`]
#' @name as_dtype
#'
#' @examplesIf pjrt::plugin_is_downloaded()
#' as_dtype("f32")
#' as_dtype("i32")
NULL

#' @rdname as_dtype
#' @importFrom stablehlo as_dtype
#' @export
stablehlo::as_dtype

#' @title Get the platform of a tensor or buffer
#'
#' @description
#' Returns the platform name (e.g. `"cpu"`, `"cuda"`) identifying
#' the compute backend.
#'
#' @details
#' Implemented via the generic [`pjrt::platform()`].
#'
#' @param x ([`tensorish`])\cr
#'   A tensor-like object.
#' @param ... Additional arguments passed to methods (unused).
#' @returns `character(1)`
#' @seealso [pjrt::platform()]
#' @name platform
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4, dtype = "f32")
#' platform(x)
NULL

#' @rdname platform
#' @importFrom pjrt platform
#' @export
pjrt::platform

#' @title Create a Shape object
#'
#' @description Constructs a `Shape` representing tensor dimensions.
#'
#' @param dims An `integer()` vector of dimension sizes (>= 0).
#' @returns A `Shape` object.
#' @seealso [shape()], [stablehlo::Shape()]
#' @name Shape
#' @rdname Shape-constructor
#' @examples
#' Shape(c(2L, 3L))
NULL

#' @rdname Shape-constructor
#' @importFrom stablehlo Shape
#' @export
stablehlo::Shape
