build_gradient_graph <- function(graph, wrt) {
  grad_env <- hashtab()

  # TODO: Handle wrt
  out_gvars <- graph@outputs

  if (length(out_gvars) != 1L) {
    cli_abort("Pullback can only be computed for functions that return a single output")
  }
  out <- out_gvars[[1L]]
  if (!identical(shape(out@aval), integer())) {
    cli_abort("Pullback can only be computed for functions that return a scalar")
  }
  # TODO: Check for float

  desc <- local_descriptor()

  # Initialize gradients for all inputs to NULL (will be set if used)
  for (input in graph@inputs) {
    grad_env[[input]] <- NULL
  }

  # TODO: Is this the right type?
  grad_env[[out]] <- maybe_box_variable(nv_scalar(1L, dtype = out@aval@dtype))
  graph

  add_or_init <- function(grad1, grad2) {
    if (is.null(grad1)) {
      return(grad2)
    }
    nvl_add(grad1, grad2)
  }

  # We need to initialize the descriptor with the forward graph's structure,
  # otherwise the nvl_<op> functions in the backward rules will extend the wrong descriptor.
  # By copying the calls and in_tree from the forward graph, we ensure that the backward
  # operations are added to the correct context.

  initialize_descriptor_from_graph(desc, graph)
  # TODO: Fix
  desc@out_tree <- graph@in_tree
  desc@outputs <- list()
  # TODO: Fix out_tree (need to adjust it if wrt is present + skip parameter inputs)
  # CONTINUE HERE

  for (call in rev(graph@calls)) {
    # here we are operating on boxes
    # TODO: wrt
    output_grads <- lapply(call@outputs, \(output) grad_env[[output]])
    input_grads <- rlang::exec(
      call@primitive[["backward"]],
      call@inputs,
      call@outputs,
      output_grads,
      !!!call@params,
      .required = rep(TRUE, length(call@inputs)) # TODO: Remove this hack
    )
    for (i in seq_along(call@inputs)) {
      input_gvar <- call@inputs[[i]]
      grad_env[[input_gvar]] <- add_or_init(grad_env[[input_gvar]], input_grads[[i]])
    }
    # here we need to determine the output.
  }

  input_grads <- lapply(graph@inputs, \(input) {
    grad <- grad_env[[input]]
    x <- if (is.null(grad)) {
      zero_const <- box(nv_constant(0L, dtype = input@aval@dtype, shape = shape(input@aval)))
      zero_const
    } else {
      grad
    }
    x@gvar
  })

  desc@outputs <- input_grads
  graph <- descriptor_to_graph(desc)

  # TODO: Need to adjust the out_tree when implementing wrt
  #graph@outputs <- lapply(graph@inputs, \(input) grad_env[[input]])
  return(graph)
}

# A non-lowering transformation builds a graph and inserts it into the parent graph.
# This is fine, because such a parent graph always exists.

#' @title Gradient
#' @description
#' Returns a function such when called, it will build up the gradient graph, insert it into its parent graph and return the outputs of the gradient graph.
#' @section Signature of the output Function:
#' ```
#' f: (Box | StaticInput).. -> (Box | StaticInput)..
#' ```
#' @export
gradient <- function(f, wrt = character()) {
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    parent_desc <- .current_descriptor()
    fwd_graph <- graphify(f, args)
    grad_graph <- build_gradient_graph(fwd_graph, wrt)
    # parent_desc is modified in place
    outputs <- inline_graph_into_desc(parent_desc, grad_graph)
    return(outputs)
  }
  formals(f_gradient) <- formals2(f)
  return(f_gradient)
}
