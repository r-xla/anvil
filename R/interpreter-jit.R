# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

#' @title JIT compile a function
#' @description
#'
#' @export
jit <- function(f) {
  cache <- hashtab()
  function(...) {
    args <- list(...)
    args_flat <- flatten(args)
    avals_in <- lapply(args_flat, \(a) raise_to_shaped(aval(a)))
    hash <- hash_shaped_tensors(avals_in)
    cache_hit <- cache[[hash]]
    if (!is.null(cache_hit)) {
      res <- rlang::exec(
        pjrt::pjrt_execute,
        cache_hit[[1L]],
        !!!args_flat,
        simplify = FALSE
      )
      res <- lapply(res, nv_tensor)
      return(unflatten(cache_hit[[2]], res))
    }

    main <- local_main(JitInterpreter)
    interpreter <- JitInterpreter(main)
    # TODO: better id
    func <- stablehlo::hlo_func(id = "main")
    func_vars <- lapply(avals_in, st2fi, func = func)
    boxes_in <- lapply(func_vars, JitBox, interpreter = interpreter)

    f_flat <- rlang::exec(flatten_fun, f, !!!args)
    outs <- rlang::exec(f_flat, !!!boxes_in)
    boxes_out <- lapply(outs[[2L]], full_raise, interpreter = interpreter)
    func_vars_out <- lapply(boxes_out, \(box) box@func_var)
    func <- do.call(stablehlo::hlo_return, func_vars_out)
    program <- pjrt_program(src = repr(func), format = "mlir")
    exec <- pjrt_compile(program)
    cache[[hash]] <- list(
      exec,
      outs[[1L]]
    )
    Recall(...)
  }
}

hash_shaped_tensors <- function(shaped_tensors) {
  digest::digest(
    lapply(shaped_tensors, \(x) list(x@shape, x@dtype)),
    algo = "xxh3_64"
  )
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
      func = stablehlo:::Func(id = stablehlo:::FuncId("main"))
    )
  )
}

method(aval, JitBox) <- function(x) {
  vt2st(x@func_var@value_type)
}
