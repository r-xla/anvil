#' @title JIT compile a function
#' @description
#' Convert a function to a JIT compiled function.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param static (`character()`)\cr
#'   Which parameters of `f` are static.
#' @param cache_size (`integer(1)`)\cr
#'   The size of the cache for the jit-compiled functions.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#'   Donated buffers can be aliased with outputs of the same type,
#'   allowing in-place operations and reducing memory usage.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device to use if the device cannot be inferred.
#' @return (`function`)
#' @export
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

    avals_in <- Map(
      function(x, is_static) {
        if (is_static) {
          x
        } else {
          if (is_anvil_tensor(x)) {
            return(nv_aten(dtype(x), shape(x), ambiguous = ambiguous(x)))
          }
          cli_abort("Expected AnvilTensor, but got {.cls {class(x)[1]}}")
        }
      },
      args_flat,
      is_static_flat
    )

    in_tree$marked <- NULL
    class(in_tree) <- c("ListNode", "Node")
    cache_hit <- cache$get(list(in_tree, avals_in, platform))
    if (!is.null(cache_hit)) {
      return(call_xla(cache_hit[[1]], cache_hit[[2]], cache_hit[[3]], args_flat, is_static_flat, cache_hit[[4]]))
    }
    compiled <- compile_to_xla(f, args_flat = avals_in, in_tree = in_tree, donate = donate, device = device)
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

  if (is.null(device)) {
    traced_devices <- unique(vapply(desc$devices, as.character, character(1)))
    if (length(traced_devices) > 1) {
      cli_abort("Tensors live on different devices: {.val {traced_devices}}.")
    }
    # TODO: Clean this up.
    # pjrt_execute should take in device instead of client
    client <- if (!length(traced_devices)) {
      pjrt::pjrt_client(Sys.getenv("PJRT_PLATFORM", "cpu"))
    } else {
      pjrt::pjrt_client(pjrt::platform(desc$devices[[1L]]))
    }
  } else {
    client <- pjrt::pjrt_client(device)
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

#' @title Compile a function to an XLA executable and wrap it as an R function
#' @description
#' Takes a function, traces it, translates to StableHLO, compiles to an XLA executable,
#' and returns an R function that executes it.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param args (`list`)\cr
#'   List of abstract input values (as passed to `f`).
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`character(1)`)\cr
#'   Target device such as `"cpu"` (default) or `"cuda"`.
#' @return (`function`)\cr
#'   A function that accepts [`AnvilTensor`] arguments (matching the flat inputs)
#'   and returns the result as [`AnvilTensor`]s.
#' @export
xla <- function(f, args, donate = character(), device = NULL) {
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

#' @title Jit an Evaluate an Expression
#' @description
#' Compiles and evaluates an expression.
#' @param expr (`expression`)\cr
#'   Expression to run.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device to use. By default (`NULL`), the device is inferred from
#'   the tensors encountered during tracing, falling back to `PJRT_PLATFORM`
#'   or `"cpu"`.
#' @return (`any`)\cr
#'   Result of the expression.
#' @export
jit_eval <- function(expr, device = NULL) {
  expr <- substitute(expr)
  eval_env <- new.env(parent = parent.frame())
  jit(\() eval(expr, envir = eval_env), device = device)()
}
