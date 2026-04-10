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
#' @param static (`character()`)\cr
#'   Names of parameters of `f` that are *not* arrays. Static values are
#'   embedded as constants in the compiled program; a new compilation is triggered whenever
#'   a static value changes. For example useful when you want R control flow in your function.
#' @param cache_size (`integer(1)`)\cr
#'   Maximum number of compiled executables to keep in the LRU cache.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#'   Donated buffers can be aliased with outputs of the same type,
#'   allowing in-place operations and reducing memory usage.
#'   An argument cannot appear in both `donate` and `static`.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device to use if it cannot be inferred from the inputs or constants.
#'   Defaults to `"cpu"`. Only supported for `backend = "xla"`.
#' @param backend (`character(1)`)\cr
#'   Compilation backend. `"xla"` (default) uses PJRT/XLA.
#'   `"quickr"` uses `quickr::quick()`. If omitted, the default comes from
#'   `default_backend()`.
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
#' local_backend("quickr")
#' h <- jit(function(x, y) x + y)
#' h(nv_array(1), nv_array(2))
jit <- function(
  f,
  static = character(),
  cache_size = 100L,
  donate = character(),
  device = NULL,
  backend = default_backend()
) {
  cache <- xlamisc::LRUCache$new(cache_size)
  backend <- assert_backend(backend)
  jit_validate_args(f, static, donate, device, backend)

  f_jit <- globals$backends[[backend]]$jit(f, static, cache, donate, device)
  formals(f_jit) <- formals2(f)
  class(f_jit) <- "JitFunction"
  attr(f_jit, "backend") <- backend
  f_jit
}

#' @export
backend.JitFunction <- function(x, ...) {
  attr(x, "backend")
}

jit_validate_args <- function(f, static, donate, device, backend) {
  assert_subset(static, formalArgs2(f))

  if (backend == "xla") {
    assert_subset(donate, formalArgs2(f))
    common <- intersect(donate, static)
    if (length(common)) {
      cli_abort("{.val {common}} cannot be both in {.arg donate} and {.arg static}.")
    }
    assert_string(device, null.ok = TRUE)
    return(invisible(NULL))
  }

  if (backend == "quickr") {
    if (length(donate)) {
      cli_abort("{.arg donate} is not supported for the {.val quickr} backend.")
    }
    if (!is.null(device)) {
      cli_abort("{.arg device} is not supported for the {.val quickr} backend.")
    }
    return(invisible(NULL))
  }

  cli_abort("Internal error: unrecognized jit backend {.val {backend}}.")
}

jit_prepare_call <- function(call, eval_env, static) {
  args <- as.list(call)[-1L]
  args <- lapply(args, eval, envir = eval_env)

  in_tree <- build_tree(mark_some(args, static))
  args_flat <- flatten(args)
  is_static_flat <- in_tree$marked
  in_tree$marked <- NULL
  class(in_tree) <- c("ListNode", "Node")

  list(
    args = args,
    args_flat = args_flat,
    is_static_flat = is_static_flat,
    in_tree = in_tree
  )
}

jit_xla_inputs <- function(args_flat, is_static_flat, device) {
  device_strs <- character()
  first_device <- NULL
  avals_in <- Map(
    function(x, is_static) {
      if (is_static) {
        return(x)
      }
      if (!is_anvil_array(x)) {
        cli_abort("Expected AnvilArray, but got {.cls {class(x)[1]}}")
      }
      if (backend(x) != "xla") {
        cli_abort("Expected {.val xla} backend, but got {.val {backend(x)}} backend.")
      }
      dev <- device(x)
      device_strs <<- c(device_strs, as.character(dev))
      if (is.null(first_device)) {
        first_device <<- dev
      }
      nv_abstract(dtype(x), shape(x), ambiguous = ambiguous(x))
    },
    args_flat,
    is_static_flat
  )

  unique_strs <- unique(device_strs)
  if (length(unique_strs) > 1) {
    cli_abort("Inputs live on different devices: {.val {unique_strs}}.")
  }

  inferred_device <- if (!is.null(first_device)) {
    first_device
  } else if (!is.null(device)) {
    device
  } else {
    NULL
  }

  list(avals_in = avals_in, device = inferred_device)
}

jit_call_xla <- function(exec, out_node, consts_flat, args_flat, is_static_flat, ambiguous_out = NULL) {
  args_nonstatic <- args_flat[!is_static_flat]
  args_unwrapped <- lapply(args_nonstatic, \(a) a$data)
  out_vals <- rlang::exec(
    pjrt::pjrt_execute,
    exec,
    !!!consts_flat,
    !!!args_unwrapped,
    simplify = FALSE
  )
  if (!is.null(ambiguous_out)) {
    out_vals <- Map(function(val, amb) nv_array(val, ambiguous = amb, backend = "xla"), out_vals, ambiguous_out)
  } else {
    out_vals <- lapply(out_vals, nv_array, backend = "xla")
  }
  unflatten(out_node, out_vals)
}

jit_xla_impl <- function(f, static, cache, donate, device) {
  function() {
    if (!is.null(.current_descriptor(silent = TRUE))) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    prep <- jit_prepare_call(match.call(), parent.frame(), static)
    inputs <- jit_xla_inputs(prep$args_flat, prep$is_static_flat, device)

    device_key <- if (!is.null(inputs$device)) as.character(inputs$device) else NULL
    cache_key <- list(prep$in_tree, inputs$avals_in, device_key)
    cache_hit <- cache$get(cache_key)
    if (!is.null(cache_hit)) {
      return(jit_call_xla(
        cache_hit[[1]],
        cache_hit[[2]],
        cache_hit[[3]],
        prep$args_flat,
        prep$is_static_flat,
        cache_hit[[4]]
      ))
    }

    compiled <- compile_to_xla(
      f,
      args_flat = inputs$avals_in,
      in_tree = prep$in_tree,
      donate = donate,
      device = inputs$device
    )
    cache$set(
      cache_key,
      list(compiled$exec, compiled$out_tree, compiled$const_arrays, compiled$ambiguous_out)
    )

    jit_call_xla(
      compiled$exec,
      compiled$out_tree,
      compiled$const_arrays,
      prep$args_flat,
      prep$is_static_flat,
      compiled$ambiguous_out
    )
  }
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

jit_quickr_impl <- function(f, static, cache) {
  function() {
    if (!is.null(.current_descriptor(silent = TRUE))) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    prep <- jit_prepare_call(match.call(), parent.frame(), static)
    inputs <- jit_quickr_inputs(prep$args_flat, prep$is_static_flat)

    cache_key <- list(prep$in_tree, inputs$avals_in, inputs$device)
    r_args <- lapply(unname(prep$args), function(a) {
      if (is_anvil_array(a)) as_array(a) else a
    })
    cache_hit <- cache$get(cache_key)
    if (!is.null(cache_hit)) {
      return(jit_call_quickr(cache_hit[[1L]], cache_hit[[2L]], cache_hit[[3L]], r_args))
    }

    compiled <- compile_to_quickr(f, args_flat = inputs$avals_in, in_tree = prep$in_tree)
    cache$set(cache_key, list(compiled$fun, compiled$out_tree, compiled$ambiguous_out))
    jit_call_quickr(compiled$fun, compiled$out_tree, compiled$ambiguous_out, r_args)
  }
}

jit_call_quickr <- function(fun, out_tree, ambiguous_out, r_args) {
  out_vals <- do.call(fun, r_args)
  out_flat <- flatten(out_vals)
  if (!is.null(ambiguous_out)) {
    out_flat <- Map(function(val, amb) nv_array(val, ambiguous = amb, backend = "quickr"), out_flat, ambiguous_out)
  } else {
    out_flat <- lapply(out_flat, nv_array, backend = "quickr")
  }
  unflatten(out_tree, out_flat)
}

#' @title Trace, lower, and compile a function to an XLA executable
#' @description
#' Takes a function, traces it into a computational graph, lowers it to StableHLO,
#' and compiles it to a PJRT executable. Returns the compiled executable along with
#' metadata needed for execution.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param args_flat (`list`)\cr
#'   Flat list of abstract input values.
#' @param in_tree (`Node`)\cr
#'   Tree structure of the inputs.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`NULL` | `character(1)`)\cr
#'   Target device (e.g. `"cpu"`, `"cuda"`). If `NULL`, inferred from traced arrays.
#' @return A `list` with elements:
#'   - `exec`: The compiled PJRT executable.
#'   - `out_tree`: The output tree structure.
#'   - `const_arrays`: Constants needed at execution time.
#'   - `ambiguous_out`: Logical vector indicating which outputs are ambiguous (`NULL` if none are).
#' @keywords internal
compile_to_xla <- function(f, args_flat, in_tree, donate = character(), device = NULL) {
  desc <- local_descriptor()
  graph <- trace_fn(f, desc = desc, toplevel = TRUE, args_flat = args_flat, in_tree = in_tree)

  # FIXME: This should also respect the devices of args_flat
  traced_devices <- unique(vapply(desc$devices, as.character, character(1)))
  if (length(traced_devices) > 1) {
    cli_abort("Tensors live on different devices: {traced_devices}.")
  }

  if (!is.null(device) && length(traced_devices) && traced_devices[[1L]] != pjrt::as_pjrt_device(device)) {
    cli_abort(
      "Provided device {.val {device}} does not match traced device {.val {traced_devices}}."
    )
  }

  # TODO: Clean this up.
  # pjrt_execute should take in device instead of client
  device <- if (!is.null(device)) {
    pjrt::as_pjrt_device(device)
  } else if (length(traced_devices)) {
    desc$devices[[1L]]
  } else {
    pjrt::as_pjrt_device(Sys.getenv("PJRT_PLATFORM", "cpu"))
  }

  graph <- inline_scalarish_constants(graph)
  graph <- remove_unused_constants(graph)

  out <- stablehlo(graph, donate = donate)
  func <- out[[1L]]
  constants <- out[[2L]]

  const_arrays <- lapply(constants, \(const) {
    if (!is_concrete_tensor(const$aval)) {
      cli_abort("Internal error: Not all constants are concrete arrays")
    }
    arr <- const$aval$data
    if (backend(arr) == "plain") {
      pjrt_buffer(as_array(arr), dtype = as.character(dtype(arr)), device = device, shape = shape(arr))
    } else {
      unwrap_if_array(arr)
    }
  })

  out_tree <- graph$out_tree

  ambiguous_out <- vapply(graph$outputs, \(x) x$aval$ambiguous, logical(1))
  if (!any(ambiguous_out)) {
    ambiguous_out <- NULL
  }

  src <- stablehlo::repr(func)
  program <- pjrt_program(src = src, format = "mlir")
  exec <- pjrt_compile(program, device = device)

  list(exec = exec, out_tree = out_tree, const_arrays = const_arrays, ambiguous_out = ambiguous_out)
}

compile_to_quickr <- function(f, args_flat, in_tree) {
  desc <- local_descriptor()
  graph <- trace_fn(f, desc = desc, toplevel = TRUE, args_flat = args_flat, in_tree = in_tree)
  ambiguous_out <- vapply(graph$outputs, \(x) x$aval$ambiguous, logical(1))
  if (!any(ambiguous_out)) {
    ambiguous_out <- NULL
  }
  list(
    fun = graph_to_quickr_function(graph),
    out_tree = graph$out_tree,
    ambiguous_out = ambiguous_out
  )
}

#' @title Ahead-of-time compile a function to XLA
#' @description
#' Compiles a function to an XLA executable via tracing.
#'
#' Returns a callable R function that executes the compiled binary.
#' Unlike [`jit()`], compilation happens eagerly at
#' definition time rather than on first call, so the input shapes and dtypes must be
#' specified upfront via abstract arrays (see [`nv_abstract()`]).
#' @details
#' Traces `f` with the given abstract `args` (via [`trace_fn()`]), lowers the resulting graph
#' via [`stablehlo()`] and then compiles it to an XLA executable via [`pjrt::pjrt_compile()`].
#' and compiles it to an XLA executable immediately.
#'
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvilArray`]s.
#' @param args (`list`)\cr
#'   List of abstract array specifications (e.g. from [`nv_abstract()`]) describing the
#'   expected shapes and dtypes of `f`'s arguments.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`character(1)`)\cr
#'   Target device such as `"cpu"` (default) or `"cuda"`.
#' @return (`function`)\cr
#'   A function that accepts [`AnvilArray`] arguments (matching the flat inputs)
#'   and returns the result as [`AnvilArray`]s.
#' @seealso [`jit()`] for lazy compilation, [`compile_to_xla()`] for the lower-level API.
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' f_compiled <- xla(function(x, y) x + y,
#'   args = list(x = nv_abstract("f32", c(2, 2)), y = nv_abstract("f32", c(2, 2)))
#' )
#' a <- nv_array(array(1:4, c(2, 2)), dtype = "f32")
#' b <- nv_array(array(5:8, c(2, 2)), dtype = "f32")
#' f_compiled(a, b)
xla <- function(f, args, donate = character(), device = NULL) {
  # FIXME: Also use device inference from trace_fn
  device <- device %||% Sys.getenv("PJRT_PLATFORM", "cpu")
  in_tree <- build_tree(args)
  args_flat <- flatten(args)
  compiled <- compile_to_xla(f, args_flat = args_flat, in_tree = in_tree, donate = donate, device = device)
  exec <- compiled$exec
  out_tree <- compiled$out_tree
  const_arrays <- compiled$const_arrays
  ambiguous_out <- compiled$ambiguous_out

  f_xla <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    args_unwrapped <- unname(lapply(args, \(a) a$data))
    out_vals <- rlang::exec(
      pjrt::pjrt_execute,
      exec,
      !!!const_arrays,
      !!!args_unwrapped,
      simplify = FALSE
    )
    if (!is.null(ambiguous_out)) {
      out_vals <- Map(function(val, amb) nv_array(val, ambiguous = amb), out_vals, ambiguous_out)
    } else {
      out_vals <- lapply(out_vals, nv_array)
    }
    unflatten(out_tree, out_vals)
  }
  formals(f_xla) <- formals2(f)
  f_xla
}

#' @title JIT-compile and evaluate an expression
#' @description
#' Convenience wrapper that JIT-compiles and immediately evaluates a single expression.
#' Equivalent to wrapping `expr` in an anonymous function, calling [`jit()`] on it, and
#' invoking the result.
#' Useful if you want to evaluate an expression once.
#' @param expr (NSE)\cr
#'   Expression to compile and evaluate.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device to use. By default (`NULL`), the device is inferred from
#'   the arrays encountered during tracing, falling back to `"cpu"`.
#'   or `"cpu"`.
#' @return (`any`)\cr
#'   Result of the compiled and evaluated expression.
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' x <- nv_array(c(1, 2, 3), dtype = "f32")
#' jit_eval(x + x)
jit_eval <- function(expr, device = NULL) {
  expr <- substitute(expr)
  eval_env <- new.env(parent = parent.frame())
  jit(\() eval(expr, envir = eval_env), device = device)()
}
