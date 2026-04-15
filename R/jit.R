#' @include backend.R
#' @title JIT compile a function
#' @description
#' Wraps a function so that it is traced and compiled on first call. Subsequent
#' calls with the same input structure, shapes, and dtypes hit an LRU cache and
#' skip recompilation. Unlike [`xla()`], the compiled executable is not created
#' eagerly but lazily on the first invocation.
#'
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvilArray`]s (and/or
#'   static arguments).
#' @param static (`character()` | `integer()`)\cr
#'   Names or positions of parameters of `f` that are *not* arrays. Static values are
#'   embedded as constants in the compiled program; a new compilation is triggered whenever
#'   a static value changes. For example useful when you want R control flow in your function.
#' @param cache_size (`integer(1)`)\cr
#'   Maximum number of compiled executables to keep in the LRU cache.
#' @param backend (`NULL` |  `character(1)`)\cr
#'   Compilation backend (e.g. `"xla"`, `"quickr"`).
#'   The special value `"auto"` defers backend selection
#'   to call time, picking the backend from the inputs (or [`default_backend()`]
#'   when there are none).
#'   If it is `NULL`, the [`default_backend()`] will be used unless the `device`
#'   is determined dynamically (`from_arg()`) in which case the device's backend
#'   will be used.
#'
#' @param device (`NULL` | `character(1)` | `PJRTDevice` | [`QuickrDevice`] | `from_arg()`)\cr
#'   Target device, forwarded to the backend-specific JIT.
#'   The default (`NULL`) uses CPU device.
#'
#'   In order to enable runtime selection of the device (useful for constant creators such as
#'   [nvl_fill()]), set to `from_arg("<device-arg>"")`.
#' @param ... Backend-specific options. Passing an option that is not supported
#'   by the selected backend raises an error. See the **XLA JIT arguments** and
#'   **Quickr JIT arguments** sections below for the options accepted by each
#'   backend.
#' @inheritSection AnvilBackendXla XLA JIT arguments
#' @inheritSection AnvilBackendQuickr Quickr JIT arguments
#' @return A `JitFunction` with the same formals as `f`.
#'   The returned wrapper expects [`AnvilArray`] inputs and returns
#'   [`AnvilArray`] values.
#' @seealso [`xla()`] for ahead-of-time compilation, [`jit_eval()`] for evaluating an expression once.
#' @return (`function`)
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' f <- jit(function(x, y) x + y)
#' f(nv_array(1), nv_array(2))
#'
#' # Static arguments enable data-dependent control flow
#' g <- jit(function(x, flag) {
#'   if (flag) x + 1 else x * 2
#' }, static = "flag")
#' g(nv_array(3), TRUE)
#' g(nv_array(3), FALSE)
#'
#' @examplesIf requireNamespace("quickr", quietly = TRUE)
#' with_backend("quickr", {
#'   h <- jit(function(x, y) x + y)
#'   h(nv_array(1), nv_array(2))
#' })
jit <- function(
  f,
  static = character(),
  cache_size = 100L,
  backend = NULL,
  device = NULL,
  ...
) {
  static <- resolve_static(f, static)
  if (inherits(device, "AnvilBackendFromArg")) {
    return(jit_auto(f, static, cache_size, device_arg = device$argname, ...))
  }
  backend_explicit <- !is.null(backend)
  if (is.null(backend)) {
    backend <- default_backend()
  }
  if (is.character(device)) {
    device <- nv_device(device, backend = backend)
  }
  if (!is.null(device)) {
    device_backend <- backend(device)
    if (backend_explicit && backend != device_backend) {
      cli_abort(
        "{.arg device} has backend {.val {device_backend}}, but {.arg backend} is {.val {backend}}."
      )
    }
    backend <- device_backend
  }
  if (backend == "auto") {
    return(jit_auto(f, static, cache_size, ...))
  }
  jit_with_backend(f, static, cache_size, backend, device = device, ...)
}

jit_with_backend <- function(f, static, cache_size, backend, ...) {
  cache <- xlamisc::LRUCache$new(cache_size)
  backend <- assert_backend(backend)
  assert_subset(static, formalArgs2(f))

  f_jit <- globals$backends[[backend]]$jit(f, static, cache, ...)
  formals(f_jit) <- formals2(f)
  class(f_jit) <- "JitFunction"
  attr(f_jit, "backend") <- backend
  f_jit
}

#' @title Select JIT device from a function argument
#' @description
#' Pass the result to [`jit()`]'s `device` argument to indicate that the
#' device should be read from a formal argument of the function being
#' compiled. At call time, the value of that argument is used to derive the
#' backend via [`backend()`] dispatch and is forwarded to the backend-specific
#' JIT as the compilation device.
#'
#' This is intended for functions that have no dynamic array inputs from which
#' the backend could otherwise be detected (e.g. array constructors like
#' [nvl_fill()] or [nvl_iota()]). If the named argument is `NULL` at call
#' time, the backend falls back to being detected from the inputs.
#'
#' @param argname (`character(1)`)\cr
#'   Name of a formal argument of the function passed to [`jit()`].
#' @return (`AnvilBackendFromArg`)\cr
#'   An object recognized by [`jit()`].
#' @seealso [`jit()`], [`backend()`]
#' @export
from_arg <- function(argname) {
  assert_string(argname)
  structure(list(argname = argname), class = "AnvilBackendFromArg")
}

resolve_static <- function(f, static) {
  if (is.integer(static)) {
    nms <- formalArgs2(f)
    if (any(static < 1L | static > length(nms))) {
      cli_abort("{.arg static} index out of range.")
    }
    return(nms[static])
  }
  static
}

#' @export
backend.JitFunction <- function(x, ...) {
  attr(x, "backend")
}

jit_auto <- function(f, static, cache_size, device_arg = NULL, ...) {
  # Lazily create per-backend jit functions
  jit_fns <- list()
  dots <- list(...)
  if (!is.null(device_arg)) {
    assert_subset(device_arg, formalArgs2(f))
  }

  wrapper <- function() {
    # Inside tracing: pass through to unwrapped function
    if (currently_tracing()) {
      cl <- match.call()
      cl[[1L]] <- f
      return(eval.parent(cl))
    }
    # Determine backend from input arrays. We avoid jit_prepare_call() here
    # because that would autoconvert inputs against an unknown backend.
    cl0 <- match.call()
    args <- lapply(as.list(cl0)[-1L], eval, envir = parent.frame())
    in_tree <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_tree$marked
    be <- if (!is.null(device_arg) && !is.null(args[[device_arg]])) {
      backend(args[[device_arg]])
    } else {
      jit_auto_detect_backend(args_flat, is_static_flat)
    }
    if (is.null(jit_fns[[be]])) {
      jit_fns[[be]] <<- do.call(
        jit_with_backend,
        c(
          list(f = f, static = static, cache_size = cache_size, backend = be),
          if (!is.null(device_arg)) list(device = from_arg(device_arg)),
          dots
        )
      )
    }
    cl0[[1L]] <- jit_fns[[be]]
    eval.parent(cl0)
  }
  formals(wrapper) <- formals2(f)
  class(wrapper) <- "JitFunction"
  attr(wrapper, "backend") <- "auto"
  wrapper
}

jit_auto_detect_backend <- function(args_flat, is_static_flat) {
  for (i in seq_along(args_flat)) {
    if (!is_static_flat[[i]] && is_anvil_array(args_flat[[i]]) && backend(args_flat[[i]]) != "plain") {
      return(backend(args_flat[[i]]))
    }
  }
  default_backend()
}

jit_prepare_call <- function(call, eval_env, static, backend) {
  args <- as.list(call)[-1L]
  args <- lapply(args, eval, envir = eval_env)

  in_tree <- build_tree(mark_some(args, static))
  args_flat <- flatten(args)
  is_static_flat <- in_tree$marked
  in_tree$marked <- NULL
  class(in_tree) <- c("ListNode", "Node")

  args_flat <- .mapply(
    function(x, is_static) if (is_static) x else autoconvert_input(x, backend),
    list(args_flat, is_static_flat),
    NULL
  )
  args <- unflatten(in_tree, args_flat)

  list(
    args = args,
    args_flat = args_flat,
    is_static_flat = is_static_flat,
    in_tree = in_tree
  )
}

autoconvert_input <- function(x, backend) {
  if (is_anvil_array(x)) {
    return(x)
  }
  if ((is.numeric(x) || is.logical(x)) && length(x) == 1L && is.null(dim(x))) {
    return(nv_scalar(x, ambiguous = TRUE, backend = backend))
  }
  if (is.array(x) && (is.numeric(x) || is.logical(x))) {
    return(nv_array(x, ambiguous = TRUE, backend = backend))
  }
  cli_abort(c(
    "Cannot autoconvert input to an {.cls AnvilArray}.",
    i = "Expected an {.cls AnvilArray}, a length-1 atomic scalar, or an {.code is.array()} value.",
    x = "Got {.cls {class(x)[1]}} of length {length(x)}."
  ))
}

jit_wrap_outputs <- function(out_flat, out_tree, ambiguous_out, backend) {
  if (!is.null(ambiguous_out)) {
    out_flat <- Map(function(val, amb) nv_array(val, ambiguous = amb, backend = backend), out_flat, ambiguous_out)
  } else {
    out_flat <- lapply(out_flat, nv_array, backend = backend)
  }
  unflatten(out_tree, out_flat)
}


#' @title JIT-compile and evaluate an expression
#' @description
#' Convenience wrapper that JIT-compiles and immediately evaluates a single expression.
#' Equivalent to wrapping `expr` in an anonymous function, calling [`jit()`] on it, and
#' invoking the result.
#' Useful if you want to evaluate an expression once.
#' @param expr (NSE)\cr
#'   Expression to compile and evaluate.
#' @param ... Backend-specific options forwarded to [`jit()`] (e.g. `device`
#'   for the `"xla"` backend, `unwrap` for the `"quickr"` backend).
#' @return (`any`)\cr
#'   Result of the compiled and evaluated expression.
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_array(c(1, 2, 3), dtype = "f32")
#' jit_eval(x + x)
jit_eval <- function(expr, ...) {
  expr <- substitute(expr)
  eval_env <- new.env(parent = parent.frame())
  jit(\() eval(expr, envir = eval_env), ...)()
}
