#' @title JIT compile a function
#' @description
#' Wraps a function so that it is traced, lowered to StableHLO, and compiled to an XLA
#' executable on first call. Subsequent calls with the same input shapes and dtypes hit an
#' LRU cache and skip recompilation. Unlike [`xla()`], the compiled executable is not
#' created eagerly but lazily on the first invocation.
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvilTensor`]s (and/or
#'   static arguments).
#' @param static (`character()`)\cr
#'   Names of parameters of `f` that are *not* tensors. Static values are
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
#'   Defaults to `"cpu"`.
#' @return A `JitFunction` with the same formals as `f`.
#' @seealso [`xla()`] for ahead-of-time compilation, [`jit_eval()`] for evaluating an expression once.
#' @return (`function`)
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' f <- jit(function(x, y) x + y)
#' f(nv_tensor(1), nv_tensor(2))
#'
#' # Static arguments enable data-dependent control flow
#' g <- jit(function(x, flag) {
#'   if (flag) x + 1 else x * 2
#' }, static = "flag")
#' g(nv_tensor(3), TRUE)
#' g(nv_tensor(3), FALSE)
jit <- function(f, static = character(), cache_size = 100L, donate = character(), device = NULL) {
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_subset(static, formalArgs2(f))
  assert_subset(donate, formalArgs2(f))
  # fmt: skip
  common <- intersect(donate, static)
  if (length(common)) {
    cli_abort("{.val {common}} cannot be both in {.arg donate} and {.arg static}.")
  }
  assert_string(device, null.ok = TRUE)

  call_xla <- function(exec, out_node, consts_flat, args_flat, is_static_flat, ambiguous_out = NULL) {
    args_nonstatic <- args_flat[!is_static_flat]
    args_unwrapped <- lapply(args_nonstatic, \(a) a$tensor)
    out_vals <- rlang::exec(
      pjrt::pjrt_execute,
      exec,
      !!!consts_flat,
      !!!args_unwrapped,
      simplify = FALSE
    )
    if (!is.null(ambiguous_out)) {
      out_vals <- Map(function(val, amb) nv_tensor(val, ambiguous = amb), out_vals, ambiguous_out)
    } else {
      out_vals <- lapply(out_vals, nv_tensor)
    }
    unflatten(out_node, out_vals)
  }

  f_jit <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())

    in_tree <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_tree$marked

    platforms <- character()
    avals_in <- Map(
      function(x, is_static) {
        if (is_static) {
          x
        } else {
          if (is_anvil_tensor(x)) {
            platforms <<- c(platforms, platform(x))
            return(nv_aten(dtype(x), shape(x), ambiguous = ambiguous(x)))
          }
          cli_abort("Expected AnvilTensor, but got {.cls {class(x)[1]}}")
        }
      },
      args_flat,
      is_static_flat
    )

    if (length(unique(platforms)) > 1) {
      cli_abort(
        "Inputs live on different platforms: {.val {unique(platforms)}}."
      )
    }
    platform <- if (length(platforms) > 0) {
      platforms[1]
    } else if (!is.null(device)) {
      device
    } else {
      # try to infer from constants
      NULL
    }

    in_tree$marked <- NULL
    class(in_tree) <- c("ListNode", "Node")
    cache_hit <- cache$get(list(in_tree, avals_in, platform))
    if (!is.null(cache_hit)) {
      return(call_xla(cache_hit[[1]], cache_hit[[2]], cache_hit[[3]], args_flat, is_static_flat, cache_hit[[4]]))
    }
    compiled <- compile_to_xla(f, args_flat = avals_in, in_tree = in_tree, donate = donate, device = platform)
    exec <- compiled$exec
    out_tree <- compiled$out_tree
    const_tensors <- compiled$const_tensors
    ambiguous_out <- compiled$ambiguous_out
    cache$set(list(in_tree, avals_in, platform), list(exec, out_tree, const_tensors, ambiguous_out))
    call_xla(exec, out_tree, const_tensors, args_flat, is_static_flat, ambiguous_out)
  }
  formals(f_jit) <- formals2(f)
  class(f_jit) <- "JitFunction"
  f_jit
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
#'   Target device (e.g. `"cpu"`, `"cuda"`). If `NULL`, inferred from traced tensors.
#' @return A `list` with elements:
#'   - `exec`: The compiled PJRT executable.
#'   - `out_tree`: The output tree structure.
#'   - `const_tensors`: Constants needed at execution time.
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
  client <- if (!is.null(device)) {
    pjrt::pjrt_client(device)
  } else if (length(traced_devices)) {
    pjrt::pjrt_client(pjrt::platform(desc$devices[[1L]]))
  } else {
    pjrt::pjrt_client(Sys.getenv("PJRT_PLATFORM", "cpu"))
  }

  graph <- inline_scalarish_constants(graph)
  graph <- remove_unused_constants(graph)

  out <- stablehlo(graph, donate = donate)
  func <- out[[1L]]
  constants <- out[[2L]]

  const_tensors <- lapply(constants, \(const) {
    if (!is_concrete_tensor(const$aval)) {
      cli_abort("Internal error: Not all constants are concrete tensors")
    }
    unwrap_if_tensor(const$aval$data)
  })

  out_tree <- graph$out_tree

  ambiguous_out <- vapply(graph$outputs, \(x) x$aval$ambiguous, logical(1))
  if (!any(ambiguous_out)) {
    ambiguous_out <- NULL
  }

  src <- stablehlo::repr(func)
  program <- pjrt_program(src = src, format = "mlir")
  exec <- pjrt_compile(program, client = client)

  list(exec = exec, out_tree = out_tree, const_tensors = const_tensors, ambiguous_out = ambiguous_out)
}

#' @title Ahead-of-time compile a function to XLA
#' @description
#' Compiles a function to an XLA executable via tracing.
#'
#' Returns a callable R function that executes the compiled binary.
#' Unlike [`jit()`], compilation happens eagerly at
#' definition time rather than on first call, so the input shapes and dtypes must be
#' specified upfront via abstract tensors (see [`nv_aten()`]).
#' @details
#' Traces `f` with the given abstract `args` (via [`trace_fn()`]), lowers the resulting graph
#' via [`stablehlo()`] and then compiles it to an XLA executable via [`pjrt::pjrt_compile()`].
#' and compiles it to an XLA executable immediately.
#'
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvilTensor`]s.
#' @param args (`list`)\cr
#'   List of abstract tensor specifications (e.g. from [`nv_aten()`]) describing the
#'   expected shapes and dtypes of `f`'s arguments.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`character(1)`)\cr
#'   Target device such as `"cpu"` (default) or `"cuda"`.
#' @return (`function`)\cr
#'   A function that accepts [`AnvilTensor`] arguments (matching the flat inputs)
#'   and returns the result as [`AnvilTensor`]s.
#' @seealso [`jit()`] for lazy compilation, [`compile_to_xla()`] for the lower-level API.
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' f_compiled <- xla(function(x, y) x + y,
#'   args = list(x = nv_aten("f32", c(2, 2)), y = nv_aten("f32", c(2, 2)))
#' )
#' a <- nv_tensor(array(1:4, c(2, 2)), dtype = "f32")
#' b <- nv_tensor(array(5:8, c(2, 2)), dtype = "f32")
#' f_compiled(a, b)
xla <- function(f, args, donate = character(), device = NULL) {
  # FIXME: Also use device inference from trace_fn
  device <- device %||% Sys.getenv("PJRT_PLATFORM", "cpu")
  in_tree <- build_tree(args)
  args_flat <- flatten(args)
  compiled <- compile_to_xla(f, args_flat = args_flat, in_tree = in_tree, donate = donate, device = device)
  exec <- compiled$exec
  out_tree <- compiled$out_tree
  const_tensors <- compiled$const_tensors
  ambiguous_out <- compiled$ambiguous_out

  f_xla <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    args_unwrapped <- unname(lapply(args, \(a) a$tensor))
    out_vals <- rlang::exec(
      pjrt::pjrt_execute,
      exec,
      !!!const_tensors,
      !!!args_unwrapped,
      simplify = FALSE
    )
    if (!is.null(ambiguous_out)) {
      out_vals <- Map(function(val, amb) nv_tensor(val, ambiguous = amb), out_vals, ambiguous_out)
    } else {
      out_vals <- lapply(out_vals, nv_tensor)
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
#'   the tensors encountered during tracing, falling back to `"cpu"`.
#'   or `"cpu"`.
#' @return (`any`)\cr
#'   Result of the compiled and evaluated expression.
#' @export
#' @examplesIf pjrt::plugin_is_downloaded("cpu")
#' x <- nv_tensor(c(1, 2, 3), dtype = "f32")
#' jit_eval(x + x, device = "cpu")
jit_eval <- function(expr, device = NULL) {
  expr <- substitute(expr)
  eval_env <- new.env(parent = parent.frame())
  jit(\() eval(expr, envir = eval_env), device = device)()
}
