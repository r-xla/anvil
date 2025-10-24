# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

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
    # TODO: Factor this out
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
      res <- rlang::exec(
        pjrt::pjrt_execute,
        cache_hit[[1L]],
        !!!args_flat[!is_static_flat],
        simplify = FALSE
      )
      res <- lapply(res, nv_tensor)
      return(unflatten(cache_hit[[2]], res))
    }

    main <- local_main(JitInterpreter)
    interpreter <- JitInterpreter(main)
    # TODO: better id
    func <- stablehlo::hlo_func(id = "main")
    boxes_in <- Map(
      function(a, static) {
        if (static) a else JitBox(st2fi(a, func), interpreter = interpreter)
      },
      avals_in,
      is_static_flat
    )
    f_flat <- rlang::exec(flatten_fun, f, in_node = in_node)
    outs <- rlang::exec(f_flat, !!!boxes_in)
    boxes_out <- lapply(outs[[2L]], full_raise, interpreter = interpreter)
    func_vars_out <- lapply(boxes_out, \(box) box@func_var)
    func <- do.call(stablehlo::hlo_return, func_vars_out)
    program <- pjrt_program(src = repr(func), format = "mlir")
    exec <- pjrt_compile(program, client = device)
    cache$set(
      avals_in,
      list(
        exec,
        outs[[1L]]
      )
    )
    Recall()
  }
  formals(f_jit) <- formals2(f)
  return(f_jit)
}

JitBox <- S7::new_class(
  "JitBox",
  parent = Box,
  properties = list(
    func_var = stablehlo::FuncVariable
  )
)

method(print, JitBox) <- function(x, ...) {
  cat(format(x), "\n")
}

method(format, JitBox) <- function(x, ...) {
  sprintf("JitBox(%s)", repr(x@func_var@value_type))
}

JitInterpreter <- S7::new_class(
  "JitInterpreter",
  parent = Interpreter,
  # TODO: Should also get a builder, once stablehlo has one
  # It definitely needs to keep track of the constants
)

method(process_primitive, JitInterpreter) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  avals_in <- lapply(boxes, \(box) box@func_var)
  avals_out <- rlang::exec(prim[["jit"]], !!!c(avals_in, params))
  lapply(avals_out, \(aval) JitBox(func_var = aval, interpreter = interpreter))
}

method(box, list(JitInterpreter, class_any)) <- function(interpreter, x) {
  # TODO: Now we have to duplicate the constant handling from the IR interpreter
  # But we can just ignore this for now
  if (inherits(x, "AnvilTensor")) {
    func_var <- stablehlo::hlo_tensor(x)
    return(JitBox(
      func_var = func_var,
      interpreter = interpreter
    ))
  }
  cli_abort("Not supported yet")
}

method(aval, JitBox) <- function(x) {
  vt2st(x@func_var@value_type)
}
