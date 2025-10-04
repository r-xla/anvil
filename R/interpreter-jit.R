# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

#' @title JIT compile a function
#' @description
#' Convert a function to a JIT compiled function.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param static (`character()`)\cr
#'   Which parameters of `f` are static.
#' @return (`function`)
#' @export
jit <- function(f, static = character()) {
  cache <- hashtab() # nolint
  assert_subset(static, formalArgs2(f))
  f_jit <- function() {
    # TODO: Factor this out
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    in_node <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_node$marked
    avals_in <- Map(
      function(a, static) {
        if (static) a else raise_to_shaped(aval(a))
      },
      args_flat,
      is_static_flat
    )
    cache_hit <- cache[[avals_in]]
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
    exec <- pjrt_compile(program)
    cache[[avals_in]] <- list(
      exec,
      outs[[1L]]
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
  avals_out <- rlang::exec(prim@jit_rule, !!!c(avals_in, params))
  lapply(avals_out, \(aval) JitBox(func_var = aval, interpreter = interpreter))
}

method(box, list(JitInterpreter, class_any)) <- function(interpreter, x) {
  # TODO: Now we have to duplicate the constant handling from the IR interpreter
  # But we can just ignore this for now
  if (inherits(x, "AnvilTensor")) {
    func_var <- stablehlo::hlo_tensor(x)
    return(JitBox(
      func_var = func_var
    ))
  }
  stop("Not supported yet")
  JitBox(
    func_var = FuncVariable(
      func = stablehlo::Func(id = stablehlo::FuncId("main"))
    )
  )
}

method(aval, JitBox) <- function(x) {
  vt2st(x@func_var@value_type)
}
