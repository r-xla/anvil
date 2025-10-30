# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

#' @title Lower a function to StableHLO
#' @description
#' Immediately lower a flattened function to a StableHLO Func object.
#' @param .f (`function`)\cr
#'   Flattened function to lower.
#' @param .avals (`list()`)\cr
#'   List of flattened abstract values (avals) for the function arguments.
#'   ShapedTensors will be traced (wrapped in HloBox), while other values
#'   will be passed through as-is (static).
#' @return (`list`) with elements:
#'   - `func`: The StableHLO `Func` object
#'   - `out_tree`: The output tree structure
#' @export
stablehlo <- function(.f, .avals) {
  main <- local_main(HloInterpreter)
  interpreter <- HloInterpreter(main)
  # TODO: better id
  func <- stablehlo::hlo_func(id = "main")
  boxes_in <- lapply(.avals, function(aval) {
    if (!inherits(aval, "anvil::ShapedTensor")) {
      # Non-ShapedTensors are static, pass through as-is
      aval
    } else {
      # ShapedTensors are traced, wrap in HloBox
      HloBox(st2fi(aval, func), interpreter = interpreter)
    }
  })

  outs <- rlang::exec(.f, !!!boxes_in)
  browser()
  boxes_out <- lapply(outs[[2L]], full_raise, interpreter = interpreter)
  func_vars_out <- lapply(boxes_out, \(box) box@func_var)
  func <- do.call(stablehlo::hlo_return, func_vars_out)

  # Return both the func and the output tree structure
  list(func = func, out_tree = outs[[1L]])
}

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
#' @return (`function`)
#' @export
jit <- function(f, static = character(), device = NULL, cache_size = 100L) {
  device <- device %??% Sys.getenv("PJRT_PLATFORM", "cpu")
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_subset(static, formalArgs2(f))
  device_str <- as.character(device)
  f_jit <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    in_node <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_node$marked
    avals_in <- Map(
      function(a, static) {
        if (static) {
          a
        } else {
          if (platform(a) != device_str) {
            cli_abort("Expected device {device_str}, but buffer has device {platform(a)}")
          }
        }
        if (static) a else raise_to_shaped(aval(a))
      },
      args_flat,
      is_static_flat
    )
    cache_hit <- cache$get(avals_in)
    # TODO: Factor this out into a function and call it at the end
    # instead of a recall, which does some work twice
    if (!is.null(cache_hit)) {
      # Only pass non-static arguments to pjrt_execute
      args_nonstatic <- args_flat[!is_static_flat]
      res <- rlang::exec(
        pjrt::pjrt_execute,
        cache_hit[[1L]],
        !!!args_nonstatic,
        simplify = FALSE
      )
      res <- lapply(res, nv_tensor)
      return(unflatten(cache_hit[[2]], res))
    }

    # Prepare flattened function and avals for lowering
    f_flat <- rlang::exec(flatten_fun, f, in_node = in_node)

    browser()
    lowered <- stablehlo(f_flat, avals_in)
    program <- pjrt_program(src = repr(lowered$func), format = "mlir")
    exec <- pjrt_compile(program)
    cache$set(avals_in, list(
      exec,
      lowered$out_tree
    ))
    Recall()
  }
  formals(f_jit) <- formals2(f)
  return(f_jit)
}

HloBox <- S7::new_class(
  "HloBox",
  parent = Box,
  properties = list(
    func_var = stablehlo::FuncVariable
  )
)

HloInterpreter <- S7::new_class(
  "HloInterpreter",
  parent = Interpreter,
  # TODO: Should also get a builder, once stablehlo has one
  # It definitely needs to keep track of the constants
)

method(process_primitive, HloInterpreter) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  avals_in <- lapply(boxes, \(box) box@func_var)
  avals_out <- rlang::exec(prim[["jit"]], !!!c(avals_in, params))
  if (!inherits(avals_out, "Special")) {
    lapply(avals_out, \(aval) HloBox(func_var = aval, interpreter = interpreter))
  } else {
    unclass(avals_out)
  }
}

method(box, list(HloInterpreter, class_any)) <- function(interpreter, x) {
  # TODO: Now we have to duplicate the constant handling from the IR interpreter
  # But we can just ignore this for now
  if (inherits(x, "AnvilTensor")) {
    func_var <- stablehlo::hlo_tensor(x)
    return(HloBox(
      func_var = func_var
    ))
  }
  cli_abort("Not supported yet")
}

method(aval, HloBox) <- function(x) {
  vt2st(x@func_var@value_type)
}
