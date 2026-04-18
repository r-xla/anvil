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
#'   `NULL` (default) uses [`default_backend()`].
#' @param device (`NULL` | `character(1)` | device object | `device_arg()`)\cr
#'   Target device. When a concrete device is specified, all arrays
#'   are moved to this device.
#'
#'   The default (`NULL`) infers the device from the inputs at call time,
#'   falling back to [`default_device()`] when there are no array inputs.
#'
#'   Devices of encountered constants are always set to the device of the (inferred) device.
#'
#'   For functions without dynamic array inputs (e.g. [nvl_fill()]) that need
#'   to work with multiple backends, use `device_arg("<argname>")` together
#'   with `backend = "auto"` to read the device from a function argument
#'   at call time and derive the backend from it.
#' @param ... Backend-specific options. Passing an option that is not supported
#'   by the selected backend raises an error. See the **XLA JIT arguments** and
#'   **Quickr JIT arguments** sections below for the options accepted by each
#'   backend.
#' @inheritSection AnvilBackendXla XLA JIT arguments
#' @inheritSection AnvilBackendQuickr Quickr JIT arguments
#'
#' @section Device and Backend selection:
#' There are various ways how to specify which device and which backend to use.
#'
#' **Concrete backend**:
#'
#' In the case where we fix a concrete backend (backend is not `"auto"`), the device can be
#' inferred during the compilation or set explicitly.
#' Setting the device explicitly allows you to enforce that the function always uses the specified
#' device, e.g. `"cuda:0"`.
#' If the `device` argument is set, all encountered arrays are converted to it.
#' E.g., when compiling a function for GPU:0, all arrays (inputs and encountered constants) will be copied to GPU:0.
#' with arrays living on the CPU, will copy the inputs to the VRAM of the GPU:0 device.
#'
#' If the device is not specified (`NULL`; default) the device will usually be inferred from the
#' inputs. If multiple devices are found, an error is thrown.
#' If no input has a device, the encountered constants device is used.
#' If there is no device either, the fallback will be to use the default device.
#'
#' **Auto backend**:
#' If the backend is set to `"auto"`
#'
#' Sometimes, there can be a function without dynamic array inputs.
#' If the backend is set to `"auto"` and no dynamic array is available to
#' TODO: Do we really need device_arg()?
#'
#' **Dynamic device selection**:
#'
#'
#'
#' @return A `JitFunction` with the same formals as `f`.
#'   The returned wrapper expects [`AnvilArray`] inputs and returns
#'   [`AnvilArray`] values.
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
  backend = NULL,
  device = NULL,
  ...
) {
  static <- resolve_arg_names(f, static, "static")
  if (is_device_arg(device)) {
    if (!(device$argname %in% static)) {
      static <- c(static, device$argname)
    }
    if (identical(backend, "auto")) {
      return(jit_auto(f, static, cache_size, device_argname = device$argname, ...))
    }
    # We use a specific backend but still determine device dynamically
    backend <- backend %||% default_backend()
    return(jit_with_backend(f, static, cache_size, backend, device = device, ...))
  }
  # device might still be NULL, which means infer from encountered arrays
  resolved <- resolve_device(device, backend)
  device <- resolved[[1L]]
  backend <- resolved[[2L]]

  if (backend == "auto") {
    return(jit_auto(f, static, cache_size, ...))
  }
  jit_with_backend(f, static, cache_size, backend, device = device, ...)
}

jit_with_backend <- function(f, static, cache_size, backend, ...) {
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_backend(backend)
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
#' [nvl_fill()] or [nvl_iota()]).
#'
#' @param argname (`character(1)`)\cr
#'   Name of a formal argument of the function passed to [`jit()`].
#' @return (`AnvilDeviceArg`)\cr
#'   An object recognized by [`jit()`].
#' @seealso [`jit()`], [`backend()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' f <- function(x) nv_scalar(1, device = x)
#' g <- jit(f, backend = "auto", device = device_arg("x"))
#' # Because backend="auto", we need to be able to infer the backend
#' # from an argument, otherwise we don't know which backend to use:
#' g(nv_device("cpu", "xla"))
device_arg <- function(argname) {
  assert_string(argname)
  structure(list(argname = argname), class = "AnvilDeviceArg")
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

jit_auto <- function(f, static, cache_size, device_argname = NULL, ...) {
  # Lazily create per-backend jit functions
  jit_fns <- list()
  dots <- list(...)
  if (!is.null(device_argname)) {
    assert_subset(device_argname, formalArgs2(f))
    # the device argument is always static
    static <- unique(c(static, device_argname))
  }

  wrapper <- function() {
    # Inside tracing: pass through to unwrapped function
    if (currently_tracing()) {
      cl <- match.call()
      cl[[1L]] <- f
      return(eval.parent(cl))
    }
    args <- lapply(as.list(match.call())[-1L], eval, envir = parent.frame())
    be <- if (!is.null(device_argname) && !is.null(args[[device_argname]])) {
      if (is.character(args[[device_argname]])) {
        default_backend()
      } else {
        backend(args[[device_argname]])
      }
    } else {
      jit_auto_detect_backend(flatten(args[!names(args) %in% static]))
    }
    if (is.null(jit_fns[[be]])) {
      jit_fns[[be]] <<- do.call(
        jit_with_backend,
        c(
          list(f = f, static = static, cache_size = cache_size, backend = be),
          if (!is.null(device_argname)) list(device = device_arg(device_argname)),
          dots
        )
      )
    }
    do.call(jit_fns[[be]], args)
  }
  formals(wrapper) <- formals2(f)
  class(wrapper) <- "JitFunction"
  attr(wrapper, "backend") <- "auto"
  wrapper
}

jit_auto_detect_backend <- function(args_flat) {
  for (i in seq_along(args_flat)) {
    if (is_anvil_array(args_flat[[i]]) && backend(args_flat[[i]]) != "plain") {
      return(backend(args_flat[[i]]))
    }
  }
  default_backend()
}

jit_prepare_call <- function(call, eval_env, static, device = NULL, backend) {
  assert_choice(backend, c("xla", "quickr"))
  args <- as.list(call)[-1L]
  args <- lapply(args, eval, envir = eval_env)

  in_tree <- build_tree(mark_some(args, static))
  args_flat <- flatten(args)
  is_static_flat <- in_tree$marked
  in_tree$marked <- NULL
  class(in_tree) <- c("ListNode", "Node")

  # Resolve device: device_arg() → extract from args, string/device → nv_device(),
  # NULL → infer from inputs, then fall back to PJRT_PLATFORM default.
  if (is_device_arg(device)) {
    device <- args[[device$argname]]
  }

  # Determine allocation device:
  # if device is specified -> use it
  # else, use first found device and if no input has device fall back to default device
  allocation_device <- if (is.null(device)) {
    found_device <- NULL
    # If any input lives on a device, use it instead
    for (i in seq_along(args_flat)) {
      if (!is_static_flat[[i]] && is_anvil_array(args_flat[[i]])) {
        found_device <- device(args_flat[[i]])
        break
      }
    }
    found_device
  } else {
    device
  }

  # whether we are copying between devices. This happens when specific device was specified in jit()
  copy_to_device <- !is.null(device)

  .mapply(
    function(x, is_static, i) if (is_static) x else check_jit_input(x, allocation_device, in_tree, i, copy_to_device),
    list(args_flat, is_static_flat, seq_along(args_flat)),
    NULL
  )

  list(
    args = args,
    args_flat = args_flat,
    is_static_flat = is_static_flat,
    in_tree = in_tree,
    device = allocation_device
  )
}

to_avals <- function(args_flat, is_static_flat) {
  Map(
    function(x, is_static) {
      if (is_static) {
        x
      } else if (is_anvil_array(x)) {
        nv_aval(dtype(x), shape(x), ambiguous(x))
      } else if ((is.numeric(x) || is.logical(x)) && length(x) == 1L && is.null(dim(x))) {
        nv_aval(default_dtype(x), integer(), ambiguous = TRUE)
      } else if (is.array(x) && (is.numeric(x) || is.logical(x))) {
        nv_aval(default_dtype(x), as.integer(dim(x)), ambiguous = TRUE)
      } else {
        cli_abort("internal error: invalid input type for jit: {.cls {class(x)[1L]}}")
      }
    },
    args_flat,
    is_static_flat
  )
}

# Check whether an input to jit is valid (w.r.t. information available before tracing)
# We don't convert yet as the concrete device is only known after tracing (respecting found constant's device)
check_jit_input <- function(x, alloc_device, in_tree = NULL, i = NULL, copy_to_device) {
  make_path <- function() {
    if (!is.null(in_tree) && !is.null(i)) tree_path(in_tree, i) else ""
  }
  if (is_anvil_array(x)) {
    if (backend(x) == "quickr") {
      return(x)
    }
    if (copy_to_device) {
      return(x)
    }

    # allocation device can be NULL if e.g. all inputs are R objects and no concrete device
    # was enforced in jit()
    if (!is.null(alloc_device) && (device(x) != alloc_device)) {
      # this can happen when there are multiple input devices but we are auto-detecting device
      path <- make_path()
      cli_abort(c(
        "Found AnvilArray input {.arg {path}} on unexpected device {device(x)}",
        i = "when using jit(f, device = NULL), ensure that all inputs live on the same device"
      ))
    }

    return(x)
  }

  if ((is.numeric(x) || is.logical(x)) && length(x) == 1L && is.null(dim(x))) {
    return(x)
  }
  if (is.array(x) && (is.numeric(x) || is.logical(x))) {
    return(x)
  }


  path <- make_path()
  msg <- if (nzchar(path)) {
    "Attempted to autoconvert {.arg {path}} to an {.cls AnvilArray}."
  } else {
    "Attempted to autoconvert input to an {.cls AnvilArray}."
  }
  cli_abort(c(
    msg,
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
