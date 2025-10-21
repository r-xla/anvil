#' @include aaa.R
#' @include tensor.R
NULL

MainInterpreter <- S7::new_class(
  "MainInterpreter",
  properties = list(
    level = class_integer,
    # TODO: Make issue in S7 with feature request
    interpreter_type = class_any, # Interpreter (class, not object)
    global_data = class_any
  )
)

globals$STACK <- list()

local_main <- function(
  interpreter_type,
  global_data = NULL,
  insert_jit = FALSE
) {
  pop <- integer()
  # TODO: probably remove insert_jit logic
  if (length(globals$STACK) == 0L) {
    pop <- 1
    if (insert_jit) {
      globals$STACK[[1L]] <- MainInterpreter(1L, JitInterpreter)
    }
  }
  pop <- c(pop, length(globals$STACK) + 1L)

  withr::defer(
    {
      globals$STACK <- globals$STACK[-pop]
    },
    envir = parent.frame()
  )

  globals$STACK[[length(globals$STACK) + 1L]] <- MainInterpreter(
    length(globals$STACK) + 1L,
    interpreter_type,
    global_data
  )
}

Interpreter <- S7::new_class(
  "Interpreter",
  properties = list(
    main = MainInterpreter
  )
)


process_primitive <- S7::new_generic(
  "process_primitive",
  "interpreter",
  function(interpreter, prim, boxes, params) {
    S7::S7_dispatch()
  }
)

# We have no pure and lift, just one box
# lifting is implemented via box(Interpreter, Box) -> Box
# and boxing via box(Interpreter, any) -> Box
box <- S7::new_generic("box", c("interpreter", "x"), function(interpreter, x) {
  S7::S7_dispatch()
})


# The object an Interpreter operates on
Box <- S7::new_class(
  "Box",
  properties = list(
    interpreter = Interpreter
  )
)

is_box <- function(x) {
  inherits(x, "anvil::Box")
}

method(aval, Box) <- function(x) {
  stop("Abstract method")
}

#' @method shape anvil::Box
#' @export
`shape.anvil::Box` <- function(x, ...) {
  shape(aval(x))
}


#' @importFrom pjrt platform
method(platform, Box) <- function(x, ...) {
  pjrt::platform(aval(x))
}

method(aval, AnvilTensor) <- function(x) {
  ConcreteTensor(x)
}

full_lower <- S7::new_generic("full_lower", "x", function(x) {
  S7::S7_dispatch()
})

method(full_lower, class_any) <- function(x) {
  x
}


# same as bind() in jax
interprete <- function(prim, args, params = list()) {
  interpreter <- current_interpreter(args)
  boxes <- lapply(args, full_raise, interpreter = interpreter)
  outs <- process_primitive(interpreter, prim, boxes, params)
  lapply(outs, full_lower)
}


full_raise <- function(interpreter, val) {
  if (!inherits(val, Box)) {
    # Closed-over constants (constants captured via lexical scoping) or
    # simply constants (like 1, 2L, ...)
    if (is_nv_type(val)) {
      return(box(interpreter, val))
    }
    # our bottom of the stack interpreter is a jit interpreter
    if (inherits(interpreter, JitInterpreter) && inherits(val, ShapedTensor)) {
      # TODO(IMPORTANT): This needs to be done properly, just a hack
      box <- JitBox(
        func_var = FuncVariable(
          stablehlo::ValueId(),
          st2vt(val),
          func = stablehlo::Func(id = stablehlo::FuncId("main"))
        ),
        interpreter = interpreter
      )
      return(box)
    }
    stop("Unsupported type: ", class(val)[1L])
  }
  level <- interpreter@main@level
  if (inherits(val@interpreter@main, S7_class(interpreter@main))) {
    # val is at the same level as top level interpreter
    return(val)
  } else if (val@interpreter@main@level < level) {
    # This can happen in nested transformations, when the inner transformation
    # captures a value that is at a lower level than the current transformation
    # function(a) {
    #   g <- grad(function(x) x + y)
    #   g(a)
    # }
    # here, the variable y is at a lower level than x, because x is also transformed via grad()
    return(box(interpreter, val))
  } else if (val@interpreter@main@level > level) {
    stop("Can't lift level ", val@interpreter@main@level, " to ", level, ".")
  } else {
    stop(
      "Different traces at same level: ",
      val@interpreter@main@level,
      " and ",
      level,
      "."
    )
  }
}

current_interpreter <- function(xs) {
  if (!length(globals$STACK)) {
    cli::cli_abort("No interpreters on the stack")
  }
  boxes <- xs[sapply(xs, inherits, Box)]
  top_main <- if (length(boxes)) {
    levels <- sapply(boxes, function(x) x@interpreter@main@level)
    boxes[which.max(levels)][[1L]]@interpreter@main
  } else {
    globals$STACK[[1]]
  }
  top_main@interpreter_type(top_main)
}

#' @method device anvil::Box
#' @export
`device.anvil::Box` <- function(x, ...) {
  device(aval(x))
}

#' @method dtype anvil::Box
#' @export
`dtype.anvil::Box` <- function(x, ...) {
  dtype(aval(x))
}
