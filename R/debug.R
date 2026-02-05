#' @include tensor.R
#' @include box.R

#' @title Debug Box Class
#' @description
#' [`AnvilBox`] subclass that wraps an [`AbstractTensor`] for use in debug mode.
#' When anvil operations (e.g. [`nv_add()`]) are called outside of [`jit()`],
#' they return `DebugBox` objects instead of actual computed results.
#' This allows checking the types and shapes of intermediate values
#' without compiling or running a computation -- see `vignette("debugging")` for details.
#'
#' The convenience constructor [`debug_box()`] creates a `DebugBox` from a dtype and shape directly.
#'
#' @section Extractors:
#' - [`dtype()`][tengen::dtype]
#' - [`shape()`][tengen::shape]
#' - [`ndims()`][tengen::ndims]
#' - [`ambiguous()`]
#'
#' @param aval ([`AbstractTensor`])\cr
#'   The abstract tensor representing the value.
#' @seealso [AnvilBox], [GraphBox], [debug_box()], [AbstractTensor]
#' @return (`DebugBox`)
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_tensor(1:4)
#' y <- nv_tensor(5:8)
#' result <- nv_add(x, y)
#' result
#' dtype(result)
#' shape(result)
#'
#' # Create directly via debug_box()
#' db <- debug_box("f32", c(2L, 3L))
#' db
#' nv_reduce_sum(db, dims = 2L)
#' @export
DebugBox <- function(aval) {
  checkmate::assert_class(aval, "AbstractTensor")

  structure(
    list(aval = aval),
    class = c("DebugBox", "AnvilBox")
  )
}

#' @export
shape.DebugBox <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.DebugBox <- function(x, ...) {
  dtype(x$aval)
}

#' @export
ambiguous.DebugBox <- function(x, ...) {
  ambiguous(x$aval)
}

#' @export
#' @method ndims DebugBox
ndims.DebugBox <- function(x, ...) {
  ndims(x$aval)
}

#' @export
print.DebugBox <- function(x, ...) {
  aval <- x$aval
  if (is_concrete_tensor(aval)) {
    cat("DebugBox(ConcreteTensor)\n")
    print(aval$data, ..., header = FALSE)
  } else if (is_literal_tensor(aval)) {
    data_str <- if (is_anvil_tensor(aval$data)) {
      trimws(capture.output(print(aval$data, ..., header = FALSE))[1L])
    } else {
      aval$data
    }
    cat(sprintf("%s:%s{%s}\n", data_str, dtype2string(aval$dtype, aval$ambiguous), shape2string(aval$shape, FALSE))) # nolint
  } else {
    cat(sprintf("%s{%s}\n", dtype2string(aval$dtype, aval$ambiguous), shape2string(aval$shape, FALSE))) # nolint
  }
  invisible(x)
}

#' @title Create a Debug Box
#' @description
#' Convenience constructor that creates a [`DebugBox`] from a data type and shape,
#' without having to manually construct an [`AbstractTensor`] first.
#'
#' @template param_dtype
#' @template param_shape
#' @template param_ambiguous
#' @seealso [DebugBox], [AbstractTensor], [trace_fn()]
#' @return ([`DebugBox`])
#' @examplesIf pjrt::plugin_is_downloaded()
#' # Create a debug box representing a 2x3 f32 tensor
#' db <- debug_box("f32", c(2L, 3L))
#' db
#' dtype(db)
#' shape(db)
#' @export
debug_box <- function(dtype, shape, ambiguous = FALSE) {
  aval <- AbstractTensor(dtype = dtype, shape = shape, ambiguous = ambiguous)
  DebugBox(aval = aval)
}

is_debug_box <- function(x) {
  inherits(x, "DebugBox")
}
