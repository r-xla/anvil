build_gradient_graph <- function(graph, wrt) {
  grad_env <- hashtab()
  required_env <- hashtab()

  out_gvars <- graph@outputs

  if (length(out_gvars) != 1L) {
    cli_abort("Pullback can only be computed for functions that return a single output")
  }
  out <- out_gvars[[1L]]
  if (!identical(shape(out@aval), integer())) {
    cli_abort("Pullback can only be computed for functions that return a scalar")
  }
  # TODO: Check for float

  # Compute which flat inputs require gradients based on wrt
  requires_grad <- requires_grad_flat(graph@in_tree, wrt)

  # Initialize required status for inputs
  for (i in seq_along(graph@inputs)) {
    required_env[[graph@inputs[[i]]]] <- requires_grad[[i]]
  }

  # Forward pass: propagate required status through the graph
  # A node requires grad if any of its inputs requires grad
  for (call in graph@calls) {
    any_input_requires <- any(vapply(call@inputs, function(x) {
      required_env[[x]] %||% FALSE # Constants don't require grad
    }, logical(1L)))

    for (out_node in call@outputs) {
      required_env[[out_node]] <- any_input_requires
    }
  }

  desc <- local_descriptor()

  # Initialize gradients for all inputs to NULL (will be set if used)
  for (input in graph@inputs) {
    grad_env[[input]] <- NULL
  }

  grad_env[[out]] <- maybe_box_variable(nv_scalar(1L, dtype = out@aval@dtype))

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
  desc@outputs <- list()

  # Backward pass
  for (call in rev(graph@calls)) {
    # Check if any input requires grad - if not, skip this call
    input_required <- vapply(call@inputs, function(x) {
      required_env[[x]] %||% FALSE
    }, logical(1L))

    if (!any(input_required)) {
      next
    }

    output_grads <- lapply(call@outputs, \(output) grad_env[[output]])
    input_grads <- rlang::exec(
      call@primitive[["backward"]],
      call@inputs,
      call@outputs,
      output_grads,
      !!!call@params,
      .required = input_required
    )
    for (i in seq_along(call@inputs)) {
      input_gvar <- call@inputs[[i]]
      grad_env[[input_gvar]] <- add_or_init(grad_env[[input_gvar]], input_grads[[i]])
    }
  }

  # Collect gradients only for inputs that require them
  input_grads <- list()
  for (i in seq_along(graph@inputs)) {
    if (!requires_grad[[i]]) {
      next
    }
    input <- graph@inputs[[i]]
    grad <- grad_env[[input]]
    x <- if (is.null(grad)) {
      maybe_box_variable(nv_constant(0L, dtype = input@aval@dtype, shape = shape(input@aval)))
    } else {
      grad
    }
    input_grads <- c(input_grads, list(x@gvar))
  }

  desc@outputs <- input_grads

  # Adjust out_tree based on wrt
  if (length(wrt)) {
    desc@out_tree <- filter_tree_by_names(graph@in_tree, wrt)
  } else {
    desc@out_tree <- graph@in_tree
  }

  graph <- descriptor_to_graph(desc)
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
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
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
