GradientTransformation <- new_class("GradientTransformation",
  parent = GraphTransformation,
  properties = list(
    wrt = class_character
  )
)

method(apply_transform, GradientTransformation) <- function(gt, args) {
  # this is the forward graph
  graph <- gt@input

  # the thing here is that we don't have to compute the forward graph anymore, because
  # we already know the graph.
  # TODO: We still have to handle `wrt`, but we ignore it for now.
  out_vars <- graph@outputs

  if (length(out_vars) != 1L) {
    cli_abort("Pullback can only be computed for functions that return a single output")
  }
  out <- out_vars[[1L]]
  if (shape(out@aval) != integer()) {
    cli_abort("Pullback can only be computed for functions that return a scalar")
  }
  # TODO: Check for float

  # TODO: Can we really modify the graph in place?
  # I don't think so, because we  might call twice
  # --> Need to create a graph copy here maybe
  graph <- clone(graph)

  # TODO: So far we don't handle constants properly, need to do this before autodiff
  # works
  grad_env[[out]] <- ConcreteTensor(nv_scalar(1L, dtype = dtype(out)))


  add_or_init <- function(grad1, grad2) {
    if (is.null(grad1)) {
      return(grad2)
    }
    nvl_add(grad1, grad2)
  }

  for (call in rev(graph@calls)) {
    required <- call@required
    output_grads <- lapply(call@outputs, \(output) grad_env[[output]])
    input_grads <- rlang::exec(call@primitive[["backward"]], call@inputs, call@outputs, output_grads, .required = call@required)
    for (i in seq_along(call@inputs)[required]) {
      input_gvar <- call@inputs[[i]]
      grad_env[[input_gvar]] <- add_or_init(grad_env[[input_gvar]], input_grads[[i]])
    }
  }
  lapply(graph@inputs, \(input) grad_env[[input]])
}

gradient <- function(f, wrt = character()) {
  gt <- GradientTransformation(f, wrt)
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())



    g <- graphify(f, args)

    grad_graph <- gradient_transform(gt, args)
    rlang::exec(nvl_graph_call, !!!args, .graph = grad_graph)
  }
  formals(f_gradient) <- formals2(f)
  return(f_gradient)
}
