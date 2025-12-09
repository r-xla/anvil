#' @title JIT compile a function
#' @description
#' Convert a function to a JIT compiled function.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param static (`character()`)\cr
#'   Which parameters of `f` are static.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device to use for the compiled function.
#'   The default (`NULL`) uses the `PJRT_PLATFORM` environment variable or defaults to "cpu".
#' @param cache_size (`integer(1)`)\cr
#'   The size of the cache for the jit-compiled functions.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#'   Donated buffers can be aliased with outputs of the same type,
#'   allowing in-place operations and reducing memory usage.
#' @return (`function`)
#' @export
jit <- function(f, static = character(), device = NULL, cache_size = 100L, donate = character()) {
  device <- device %??% Sys.getenv("PJRT_PLATFORM", "cpu")
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_subset(static, formalArgs2(f))
  device_str <- as.character(device)

  call_xla <- function(exec, out_node, consts_flat, args_flat, is_static_flat) {
    args_nonstatic <- args_flat[!is_static_flat]
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

    avals_in <- Map(
      function(x, is_static) {
        if (is_static) {
          x
        } else {
          if (platform(x) != device_str) {
            cli_abort("Expected device {device_str}, but buffer has device {platform(x)}")
          }
          raise_to_shaped(aval(x))
        }
      },
      args_flat,
      is_static_flat
    )

    cache_hit <- cache$get(avals_in)
    if (!is.null(cache_hit)) {
      return(call_xla(cache_hit[[1]], cache_hit[[2]], cache_hit[[3]], args_flat, is_static_flat))
    }
    # TODO: Give trace_fn() argument in_tree, so we don't have to do this twice
    graph <- trace_fn(f, args)

    out <- stablehlo(graph, donate = donate)
    func <- out[[1L]]
    constants <- out[[2L]]

    const_tensors <- lapply(constants, \(const) {
      if (!is_concrete_tensor(const@aval)) {
        cli_abort("Internal error: Not all constants are concrete tensors")
      }
      const@aval@data
    })

    out_tree <- graph@out_tree
    src <- stablehlo::repr(func)
    program <- pjrt_program(src = src, format = "mlir")
    exec <- pjrt_compile(program)
    cache$set(avals_in, list(exec, out_tree, const_tensors))
    call_xla(exec, out_tree, const_tensors, args_flat, is_static_flat)
  }
  formals(f_jit) <- formals2(f)
  f_jit
}
