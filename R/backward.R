transform_gradient <- function(graph, wrt) {
  grad_env <- hashtab()
  required_env <- hashtab()

  out_gvals <- graph@outputs

  if (length(out_gvals) != 1L) {
    cli_abort("gradient can only be computed for functions that return a single output")
  }
  out <- out_gvals[[1L]]
  if (!identical(shape(out@aval), integer())) {
    cli_abort("gradient can only be computed for functions that return a scalar")
  }
  dt <- out@aval@dtype
  if (!(dt == dtype("f32") || dt == dtype("f64"))) {
    cli_abort(c(
      x = "gradient can only be computed for functions that return float scalar",
      i = "Got dtype={.field {repr(dt)}}"
    ))
  }

  requires_grad <- flat_mask_from_names(graph@in_tree, wrt)

  for (i in seq_along(graph@inputs)) {
    required_env[[graph@inputs[[i]]]] <- requires_grad[[i]]
  }
  for (i in seq_along(graph@constants)) {
    required_env[[graph@constants[[i]]]] <- FALSE
  }

  # Forward pass: propagate required status through the graph
  # A node requires grad if any of its inputs requires grad
  for (call in graph@calls) {
    any_input_requires <- any(vapply(
      call@inputs,
      function(x) {
        if (is_graph_literal(x)) {
          return(FALSE)
        }
        required_env[[x]]
      },
      logical(1L)
    ))

    for (out_node in call@outputs) {
      required_env[[out_node]] <- any_input_requires
    }
  }

  desc <- local_descriptor()

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
  init_desc_from_graph(desc, graph, outputs = FALSE)
  grad_env[[out]] <- get_box_or_register_const(desc, nv_scalar(1L, dtype = out@aval@dtype))

  # Backward pass
  for (call in rev(graph@calls)) {
    # Check if any input requires grad - if not, skip this call
    input_required <- vapply(
      call@inputs,
      function(x) {
        required_env[[x]] %||% FALSE
      },
      logical(1L)
    )

    if (!any(input_required)) {
      next
    }

    output_grads <- lapply(call@outputs, \(output) {
      grad <- grad_env[[output]]
      if (is.null(grad)) {
        nvl_full(0L, dtype = dtype(output), shape = shape(output))
      } else {
        grad
      }
    })

    # Skip if all output gradients are NULL (dead code)
    if (all(vapply(output_grads, is.null, logical(1L)))) {
      next
    }

    input_grads <- rlang::exec(
      call@primitive[["backward"]],
      call@inputs,
      call@outputs,
      output_grads,
      !!!call@params,
      .required = input_required
    )
    # input_grads[!input_required] is list of NULLs
    for (i in seq_along(call@inputs)) {
      input_gval <- call@inputs[[i]]
      grad_env[[input_gval]] <- add_or_init(grad_env[[input_gval]], input_grads[[i]])
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
      const <- get_box_or_register_const(desc, nv_scalar(0L, dtype = input@aval@dtype))
      nv_broadcast_to(const, shape(input@aval))
    } else {
      grad
    }
    # browser()
    input_grads <- c(input_grads, list(x@gnode))
  }

  desc@outputs <- input_grads

  # Adjust out_tree based on wrt
  desc@out_tree <- if (length(wrt)) {
    filter_list_node(graph@in_tree, wrt)
  } else {
    graph@in_tree
  }

  graph <- descriptor_to_graph(desc)
  return(graph)
}


# A non-lowering transformation builds a graph and inserts it into the parent graph.
# This is fine, because such a parent graph always exists.

#' @title Gradient
#' @description
#' Transform a function to its gradient.
#' @param f (`function`)\cr
#'   Function to compute the gradient of.
#' @param wrt (`character`)\cr
#'   Names of the arguments to compute the gradient with respect to.
#' @export
gradient <- function(f, wrt = NULL) {
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    parent_desc <- .current_descriptor()
    fwd_graph <- trace_fn(f, args)
    grad_graph <- transform_gradient(fwd_graph, wrt)
    # parent_desc is modified in place
    outputs <- inline_graph_into_desc(parent_desc, grad_graph)
    return(outputs)
  }
  formals(f_gradient) <- formals2(f)
  return(f_gradient)
}
