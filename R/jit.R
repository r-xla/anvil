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
#' @param backend (`character(1)`)\cr
#'   Compilation backend. `"xla"` (default) uses PJRT/XLA.
#'   `"quickr"` uses `quickr::quick()`. If omitted, the default comes from
#'   `default_backend()`.
#' @param ... Backend-specific options. Passing an option that is not supported
#'   by the selected backend raises an error. See the **XLA JIT arguments** and
#'   **Quickr JIT arguments** sections below for the options accepted by each
#'   backend.
#' @inheritSection AnvilBackendXla XLA JIT arguments
#' @inheritSection AnvilBackendQuickr Quickr JIT arguments
#' @return A `JitFunction` with the same formals as `f`.
#'   The returned wrapper expects [`AnvilArray`] inputs and returns
#'   [`AnvilArray`] values (unless `unwrap = TRUE` is passed to the
#'   `"quickr"` backend).
#' @seealso [`xla()`] for ahead-of-time compilation, [`jit_eval()`] for evaluating an expression once.
#' @return (`function`)
#' @export
#' @examplesIf pjrt::plugins_downloaded()
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
  backend = default_backend(),
  ...
) {
  static <- resolve_static(f, static)
  cache <- xlamisc::LRUCache$new(cache_size)
  backend <- assert_backend(backend)
  assert_subset(static, formalArgs2(f))

  f_jit <- globals$backends[[backend]]$jit(f, static, cache, ...)
  formals(f_jit) <- formals2(f)
  class(f_jit) <- "JitFunction"
  attr(f_jit, "backend") <- backend
  f_jit
}

resolve_static <- function(f, static) {
  resolve_arg_names(f, static, "static")
}

# Translate a character-or-integer argument selector into character names
# of the formals of `f`. `arg` is used in error messages. Rejects `"..."`
# (whether supplied by name or resolved from an integer position), since it
# does not name a single argument.
resolve_arg_names <- function(f, x, arg) {
  if (is.null(x)) {
    return(x)
  }
  if (is.integer(x)) {
    nms <- formalArgs2(f)
    if (any(x < 1L | x > length(nms))) {
      cli_abort("{.arg {arg}} index out of range.")
    }
    x <- nms[x]
  }
  if ("..." %in% x) {
    cli_abort("{.arg {arg}} must not contain {.val ...}.")
  }
  x
}

#' @export
backend.JitFunction <- function(x, ...) {
  attr(x, "backend")
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
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3), dtype = "f32")
#' jit_eval(x + x)
jit_eval <- function(expr, ...) {
  expr <- substitute(expr)
  eval_env <- new.env(parent = parent.frame())
  jit(\() eval(expr, envir = eval_env), ...)()
}
