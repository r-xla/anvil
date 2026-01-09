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
#'   The device to use if no input tensors are provided to infer the platform.
#' @return (`function`)
#' @export
jit <- function(f, static = character(), cache_size = 100L, donate = character(), device = NULL) {
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_subset(static, formalArgs2(f))
  assert_string(device, null.ok = TRUE)

  call_xla <- function(exec, out_node, consts_flat, args_flat, is_static_flat, platform) {
    args_nonstatic <- args_flat[!is_static_flat]
    args_nonstatic <- lapply(args_nonstatic, function(arg) {
      if (test_scalar(arg)) {
        return(nv_scalar(arg, device = platform))
      }
      arg
    })
    out_vals <- rlang::exec(
      pjrt::pjrt_execute,
      exec,
      !!!consts_flat,
      !!!args_nonstatic,
      simplify = FALSE
    )
    out_vals <- lapply(out_vals, nv_tensor)
    unflatten(out_node, out_vals)
  }

  f_jit <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())

    in_node <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_node$marked

    platforms <- character()
    avals_in <- Map(
      function(x, is_static) {
        if (is_static) {
          return(x)
        }
        if (test_scalar(x) && (is.numeric(x) || is.logical(x))) {
          return(nv_aten(default_dtype(x), integer()))
        }
        if (!is_anvil_tensor(x)) {
          cli_abort("Expected anvil tensor, but got {.cls {class(x)[1]}}")
        }
        platforms <<- c(platforms, platform(x))
        nv_aten(dtype(x), shape(x))
      },
      args_flat,
      is_static_flat
    )

    if (length(unique(platforms)) > 1) {
      cli_abort(
        "Inputs live on different platforms: {.val {unique(platforms)}}."
      )
    }
    # FIXME: platform does not always return "cuda" on CUDA gpus,
    # so we might store the same entry twice (via "cuda" and via the specific GPU-dependent name)
    platform <- if (length(platforms) > 0) {
      platforms[1]
    } else if (!is.null(device)) {
      device
    } else {
      Sys.getenv("PJRT_PLATFORM", "cpu")
    }

    cache_hit <- cache$get(list(avals_in, platform))
    if (!is.null(cache_hit)) {
      return(call_xla(cache_hit[[1]], cache_hit[[2]], cache_hit[[3]], args_flat, is_static_flat, platform))
    }
    desc <- local_descriptor()
    in_tree <- in_node
    in_tree$marked <- NULL
    class(in_tree) <- c("ListNode", "Node")
    graph <- trace_fn(f, desc = desc, toplevel = TRUE, flat_inputs = avals_in, in_tree = in_tree)
    graph <- inline_scalarish_constants(graph)
    graph <- remove_unused_constants(graph)

    out <- stablehlo(graph, donate = donate)
    func <- out[[1L]]
    constants <- out[[2L]]

    const_tensors <- lapply(constants, \(const) {
      if (!is_concrete_tensor(const$aval)) {
        cli_abort("Internal error: Not all constants are concrete tensors")
      }
      const$aval$data
    })

    out_tree <- graph$out_tree
    src <- stablehlo::repr(func)
    program <- pjrt_program(src = src, format = "mlir")
    exec <- pjrt_compile(program, client = pjrt::pjrt_client(platform))
    cache$set(list(avals_in, platform), list(exec, out_tree, const_tensors))
    call_xla(exec, out_tree, const_tensors, args_flat, is_static_flat, platform)
  }
  formals(f_jit) <- formals2(f)
  f_jit
}

#' @title Jit an Evaluate an Expression
#' @description
#' Compiles and evaluates an expression.
#' @param expr (`expression`)\cr
#'   Expression to run.
#' @return (`any`)\cr
#'   Result of the expression.
#' @export
jit_eval <- function(expr) {
  expr <- substitute(expr)
  env <- parent.frame()
  jit(\() eval(expr, envir = env))()
}
