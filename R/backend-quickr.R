#' @include backend.R
NULL

#' @title Quickr device
#' @description
#' Device descriptor for the quickr backend. The only supported `type` is
#' `"cpu"`.
#' @param x (`character(1)`)\cr
#'   Device type. Currently only supports `"cpu"`.
#' @return A `QuickrDevice` object.
#' @seealso [`nv_device()`], [`AnvilBackendQuickr()`].
#' @export
quickr_device <- function(x = "cpu") {
  assert_choice(x, c("cpu"))
  structure(list(device = x), class = "QuickrDevice")
}

#' @export
as.character.QuickrDevice <- function(x, ...) x$device

#' @export
format.QuickrDevice <- function(x, ...) paste0("QuickrDevice(", x$device, ")")

#' @export
print.QuickrDevice <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

quickr_jit_aval <- function(x) {
  if (!is_anvil_array(x)) {
    cli_abort("Expected AnvilArray, but got {.cls {class(x)[1]}}")
  }
  if (backend(x) != "quickr") {
    cli_abort("Expected {.val quickr} backend, but got {.val {backend(x)}} backend.")
  }
  nv_abstract(dtype(x), shape(x), ambiguous = ambiguous(x))
}

jit_quickr_inputs <- function(args_flat, is_static_flat) {
  avals_in <- Map(
    function(x, is_static) {
      if (is_static) {
        return(x)
      }
      quickr_jit_aval(x)
    },
    args_flat,
    is_static_flat
  )

  list(avals_in = avals_in, device = NULL)
}

jit_quickr_impl <- function(f, static, cache, unwrap) {
  function() {
    # calling a jitted function within another jitted function --> re-trace the original closure
    if (currently_tracing()) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    prep <- jit_prepare_call(match.call(), parent.frame(), static, "quickr")
    inputs <- jit_quickr_inputs(prep$args_flat, prep$is_static_flat)

    cache_key <- list(prep$in_tree, inputs$avals_in, inputs$device)
    r_args_flat <- lapply(prep$args_flat, function(a) {
      if (is_anvil_array(a)) as_array(a) else a
    })
    cache_hit <- cache$get(cache_key)
    if (!is.null(cache_hit)) {
      return(cache_hit(r_args_flat))
    }

    compiled <- compile_to_quickr(f, args_flat = inputs$avals_in, in_tree = prep$in_tree, unwrap = unwrap, flat = TRUE)
    cache$set(cache_key, compiled$fun)
    compiled$fun(r_args_flat)
  }
}

compile_to_quickr <- function(f, args_flat, in_tree, unwrap = FALSE, flat = FALSE) {
  desc <- local_descriptor()
  graph <- trace_fn(f, desc = desc, toplevel = TRUE, args_flat = args_flat, in_tree = in_tree)
  list(fun = graph_to_quickr_function(graph, unwrap = unwrap, flat = flat))
}

#' Quickr backend
#'
#' Constructs the quickr backend, which stores array data as plain R arrays and
#' compiles jitted functions to R code via the
#' [quickr](https://CRAN.R-project.org/package=quickr) package.
#'
#' To use it, the `"quickr"` package needs to be installed.
#'
#' Registered automatically under the name `"quickr"` when the package is
#' loaded; call [`local_backend("quickr")`][local_backend()] or
#' [`with_backend("quickr", ...)`][with_backend()] to use it. Requires the
#' quickr package to be installed.
#'
#' @section Data representation:
#' An [`AnvilArray`] with `backend = "quickr"` is, under the hood, a plain R
#' vector or array (`numeric`, `integer`, or `logical`) stored in the `$data`
#' field. [`as_array()`] returns the underlying vector/array directly without
#' copying, and [`nv_array()`] simply wraps an R vector/array. As a
#' consequence, there is no separate notion of a device: data always lives in
#' R's memory and computation always runs on the CPU.
#'
#' @section Status:
#' This backend is **experimental** and has a number of limitations:
#'
#' * Compilation (tracing + quickr lowering) is somewhat slow, so it is best
#'   suited to long-running or repeatedly-called functions where the one-time
#'   compilation cost is amortized.
#' * Only a subset of the primitives that the XLA backend supports are currently
#'   lowered to quickr code. See `vignette("primitives")` for an overview.
#' * Only the data types `f64`, `i32`, and `bool` are supported.
#' * Only CPU execution is supported.
#'
#' @section Quickr JIT arguments:
#'
#' * `unwrap` (`logical(1)`, default `FALSE`): if `TRUE`, the compiled function
#'   returns plain R arrays instead of [`AnvilArray`]s. Useful when the jitted
#'   function's output is consumed by non-anvil R code and the extra wrapping
#'   would only get stripped again.
#'
#' @return An [`AnvilBackend`] object with subclass `"AnvilBackendQuickr"`.
#' @seealso [`AnvilBackend()`], [`AnvilBackendXla()`], [`local_backend()`], [`jit()`].
#' @export
AnvilBackendQuickr <- function() {
  backend <- AnvilBackend(
    data_constructor = function(data, dtype, shape, device, ambiguous) {
      if (is.null(dtype)) {
        dtype <- if (is.double(data)) FloatType(64) else default_dtype(data)
      }
      if (!is_dtype(dtype)) {
        dtype <- as_dtype(dtype)
      }
      if (is.null(shape)) {
        shape <- if (!is.null(dim(data))) {
          as.integer(dim(data))
        } else if (length(data) == 1L) {
          1L
        } else {
          as.integer(length(data))
        }
      }
      dtype_chr <- as.character(dtype)
      data <- switch(
        substr(dtype_chr, 1, 1),
        "f" = as.double(data),
        "i" = ,
        "u" = as.integer(data),
        "b" = as.logical(data),
        as.double(data)
      )
      if (length(shape) >= 1L) {
        dim(data) <- shape
      }
      structure(
        list(data = data, dtype = dtype, shape = shape, ambiguous = ambiguous, backend = "quickr"),
        class = "AnvilArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    ambiguous = function(x) x$ambiguous,
    as_array = function(x) x$data,
    as_raw = function(x, row_major) as.raw(x$data),
    platform = function(x) "cpu",
    device = function(x) quickr_device("cpu"),
    new_device = function(x) quickr_device(x),
    print_data = function(x, footer) {
      print(x$data)
      cat(footer, "\n")
    },
    jit = function(f, static, cache, unwrap = FALSE) {
      assert_flag(unwrap)
      jit_quickr_impl(f, static, cache, unwrap)
    }
  )
  class(backend) <- c("AnvilBackendQuickr", class(backend))
  backend
}

register_backend("quickr", AnvilBackendQuickr())
