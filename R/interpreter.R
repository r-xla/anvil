#' @include list_of.R
#' @include aval.R
NULL

MainInterpreter <- S7::new_class(
  "MainInterpreter",
  properties = list(
    level = class_integer,
    interpreter_type = class_any # Interpreter (class, not object)
    # We have no global data, as we attach it to the tracers.
  )
)

globals$STACK <- list()

local_main <- function(interpreter_type) {
  withr::defer({
    globals$STACK <- globals$STACK[-length(globals$STACK)]
  })
  globals$STACK[[length(globals$STACK) + 1L]] <- MainInterpreter(
    length(globals$STACK) + 1L,
    interpreter_type
  )
}

Interpreter <- S7::new_class("Interpreter",
  properties = list(
    main = MainInterpreter
  )
)


process_primitive <- S7::new_generic(
  "process_primitive",
  "interpreter",
  function(interpreter, prim, args, params) {
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
Box <- S7::new_class("Box",
  properties = list(
    interpreter = Interpreter
  )
)
Boxes <- new_list_of("Boxes", Box)

method(aval, Box) <- function(x) {
  stop("Abstract method")
}

method(aval, class_any) <- function(x) {
  if (is_nvl_type(box)) {
    ConcreteArray(nvl_array(x))
  } else {
    stop("Type has no aval")
  }
}

full_lower <- S7::new_generic("full_lower", "x", function(x) {
  S7::S7_dispatch()
})

method(full_lower, class_any) <- function(x) {
  x
}


# same as bind() in jax
interprete <- function(prim, args, params = list()) {
  interpreter <- find_top_interpreter(args)
  boxes <- lapply(args, \(arg) full_raise(interpreter, arg))
  outs <- process_primitive(interpreter, prim, boxes, params)
  lapply(outs, full_lower)
}



full_raise <- function(interpreter, val) {
  if (!inherits(val, Box)) {
    if (is_nvl_type(val)) {
      return(box(val))
    }
    stop("Unsupported type: ", class(val)[1L])
  }
  level <- interpreter@main@level
  if (inherits(val@.interpreter@main, interpreter@main)) {
  # val is at the same level as top level interpreter
    return(val)
  } else if (val@.interpreter@main@level < level) {
    # This can happen in nested transformations, when the inner transformation
    # captures a value that is at a lower level than the current transformation
    # function(a) {
    #   g <- grad(function(x) x + y)
    #   g(a)
    # }
    # here, the variable y is at a lower level than x, because x is also transformed via grad()
    return(box(interpreter, val))
  } else if (val@.interpreter@main@level > level) {
    stop("Can't lift level ", val@.interpreter@main@level, " to ", level, ".")
  } else {
    stop("Different traces at same level: ", val@.interpreter@main@level, " and ", level, ".")
  }
}

find_top_interpreter <- function(xs) {
  boxes <- xs[sapply(xs, inherits, Box)]
  top_main <- if (length(boxes)) {
    levels <- sapply(boxes, function(x) x@interpreter@level)
    top_main <- boxes[which.max(levels)][[1L]]@interpreter
  } else {
    top_main <- globals$STACK[[1]]
  }
  top_main@interpreter_type(top_main)
}
