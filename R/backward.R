check_wrt_tensorish <- function(args_flat, is_wrt_flat) {
  for (i in seq_along(args_flat)) {
    if (is_wrt_flat[[i]] && !is_tensorish(args_flat[[i]], literal = FALSE)) {
      if (!is_tensorish(args_flat[[i]], literal = TRUE)) {
        cli_abort(c(
          "Cannot compute gradient with respect to non-tensor argument.",
          x = "Got {.cls {class(args_flat[[i]])}}"
        ))
      }
      if (!inherits(dtype_abstract(args_flat[[i]]), "FloatType")) {
        cli_abort(c(
          "Can only compute gradient with respect to float tensors.",
          x = "Got {repr(dtype_abstract(args_flat[[i]]))}"
        ))
      }
    }
  }
}

prepare_gradient_args <- function(args, wrt) {
  args_flat <- flatten(args)
  if (!is.null(wrt)) {
    in_tree <- build_tree(mark_some(args, wrt))
    is_wrt_flat <- in_tree$marked
    in_tree$marked <- NULL
    class(in_tree) <- c("ListNode", "Node")
  } else {
    in_tree <- build_tree(args)
    is_wrt_flat <- rep(TRUE, length(args_flat))
  }
  check_wrt_tensorish(args_flat, is_wrt_flat)
  list(args_flat = args_flat, in_tree = in_tree)
}

#' @title Transform a graph to its gradient
#' @description
#' Low-level graph transformation that appends the backward pass to a
#' traced [`AnvilGraph`]. The function `f` represented by `graph` must return a single
#' float scalar. The resulting graph computes the gradients of that scalar with respect
#' to the inputs specified by `wrt`.
#'
#' The backward rules are stored in `$rules[["backward"]]` of the primitives.
#'
#' This is the building block used by [`gradient()`] and [`value_and_gradient()`]; prefer
#' those higher-level wrappers unless you need to operate on graphs directly.
#' @param graph ([`AnvilGraph`])\cr
#'   The graph to transform. Must produce a single scalar float output.
#' @param wrt (`character`)\cr
#'   Names of the graph inputs to differentiate with respect to.
#' @return An [`AnvilGraph`] whose outputs are the requested gradients.
#' @seealso [`gradient()`], [`value_and_gradient()`]
#' @export
#' @examples
#' graph <- trace_fn(nvl_mul, list(nv_aten("f32", c()), nv_aten("f32", c())))
#' graph
#' transform_gradient(graph, "lhs")
transform_gradient <- function(graph, wrt) {
  grad_env <- hashtab()
  required_env <- hashtab()

  out_gvals <- graph$outputs

  if (length(out_gvals) != 1L) {
    cli_abort("gradient can only be computed for functions that return a single output")
  }
  out <- out_gvals[[1L]]
  if (!identical(shape(out$aval), integer())) {
    cli_abort("gradient can only be computed for functions that return a scalar")
  }
  dt <- out$aval$dtype
  if (!(dt == as_dtype("f32") || dt == as_dtype("f64"))) {
    cli_abort(c(
      x = "gradient can only be computed for functions that return float scalar",
      i = "Got dtype={.field {repr(dt)}}"
    ))
  }

  requires_grad_all <- flat_mask_from_names(graph$in_tree, wrt)
  # Filter mask to only include entries for non-static inputs
  # Note that in `gradient()` we already check that non of the wrts can be static.
  # since in_tree may include static (non-tensor) args not present in graph$inputs.
  is_static <- graph$is_static_flat
  requires_grad <- if (is.null(is_static)) {
    requires_grad_all
  } else {
    requires_grad_all[!is_static]
  }

  for (i in seq_along(graph$inputs)) {
    required_env[[graph$inputs[[i]]]] <- requires_grad[[i]]
  }
  for (i in seq_along(graph$constants)) {
    required_env[[graph$constants[[i]]]] <- FALSE
  }

  # Forward pass: propagate required status through the graph
  # A node requires grad if any of its inputs requires grad
  for (call in graph$calls) {
    any_input_requires <- any(vapply(
      call$inputs,
      function(x) {
        if (is_graph_literal(x)) {
          # literals don't depend on anything (they are either inlined constants or literals)
          return(FALSE)
        }
        required_env[[x]]
      },
      logical(1L)
    ))

    for (out_node in call$outputs) {
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
  # including the forward calls, so that intermediate values are available
  # for the backward rules.
  init_desc_from_graph(desc, graph, outputs = FALSE)
  grad_env[[out]] <- get_box_or_register_const(
    desc,
    nv_scalar(1L, dtype = out$aval$dtype, ambiguous = out$aval$ambiguous)
  )

  # Backward pass
  for (call in rev(graph$calls)) {
    # Check if any input requires grad - if not, skip this call
    input_required <- vapply(
      call$inputs,
      function(x) {
        required_env[[x]] %||% FALSE
      },
      logical(1L)
    )

    if (!any(input_required)) {
      next
    }

    output_grads <- lapply(call$outputs, \(output) {
      grad <- grad_env[[output]]
      if (is.null(grad)) {
        # output grad might be NULL if there is dead code
        nvl_fill(0L, dtype = dtype(output), shape = shape(output))
      } else {
        grad
      }
    })

    # Convert gvals to boxes for the backward pass
    input_boxes <- lapply(call$inputs, \(x) desc$gval_to_box[[x]])
    output_boxes <- lapply(call$outputs, \(x) desc$gval_to_box[[x]])

    input_grads <- rlang::exec(
      call$primitive[["backward"]],
      input_boxes,
      output_boxes,
      output_grads,
      !!!call$params,
      .required = input_required
    )
    # input_grads[!input_required] is list of NULLs
    for (i in seq_along(call$inputs)) {
      input_gval <- call$inputs[[i]]
      grad_env[[input_gval]] <- add_or_init(grad_env[[input_gval]], input_grads[[i]])
    }
  }

  # Collect gradients only for inputs that require them
  input_grads <- list()
  for (i in seq_along(graph$inputs)) {
    if (!requires_grad[[i]]) {
      next
    }
    input <- graph$inputs[[i]]
    grad <- grad_env[[input]]
    x <- if (is.null(grad)) {
      const <- get_box_or_register_const(
        desc,
        nv_scalar(0L, dtype = input$aval$dtype, ambiguous = input$aval$ambiguous)
      )
      nv_broadcast_to(const, shape(input$aval))
    } else {
      grad
    }
    # ensure ambiguity of gradient is the same as the input
    x$gnode$aval$ambiguous <- input$aval$ambiguous
    input_grads <- c(input_grads, list(x$gnode))
  }

  desc$outputs <- input_grads

  # Adjust out_tree based on wrt
  desc$out_tree <- if (length(wrt)) {
    filter_list_node(graph$in_tree, wrt)
  } else {
    graph$in_tree
  }

  graph <- descriptor_to_graph(desc)
  return(graph)
}


# A non-lowering transformation builds a graph and inserts it into the parent graph.
# This is fine, because such a parent graph always exists.

#' @title Gradient
#' @description
#' Returns a new function that computes the gradient of `f` via reverse-mode automatic
#' differentiation. `f` must return a single float scalar. The returned function has the
#' same signature as `f` and returns the gradients in the same structure as the inputs
#' (or the subset selected by `wrt`).
#' @param f (`function`)\cr
#'   Function to differentiate. Arguments can be tensorish ([`AnvilTensor`]) or
#'   static (non-tensor) values. Must return a single scalar float tensor.
#' @param wrt (`character` or `NULL`)\cr
#'   Names of the arguments to compute the gradient with respect to.
#'   Only tensorish (float tensor) arguments can be included; static arguments
#'   must not appear in `wrt`.
#'   If `NULL` (the default), the gradient is computed with respect to all
#'   arguments (which must all be tensorish in that case).
#' @return `function`
#' @seealso [`value_and_gradient()`] to get both the output and gradients,
#'   [`transform_gradient()`] for the low-level graph transformation.
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' f <- function(x, y) sum(x * y)
#' g <- jit(gradient(f))
#' g(nv_tensor(c(1, 2), dtype = "f32"), nv_tensor(c(3, 4), dtype = "f32"))
#'
#' # Differentiate with respect to a single argument
#' g_x <- jit(gradient(f, wrt = "x"))
#' g_x(nv_tensor(c(1, 2), dtype = "f32"), nv_tensor(c(3, 4), dtype = "f32"))
#'
#' # Static (non-tensor) arguments are passed through but cannot be in wrt
#' f2 <- function(x, power) sum(x^power)
#' g2 <- jit(gradient(f2, wrt = "x"), static = "power")
#' g2(nv_tensor(c(1, 2, 3), dtype = "f32"), power = 2L)
gradient <- function(f, wrt = NULL) {
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    prep <- prepare_gradient_args(args, wrt)

    parent_desc <- .current_descriptor(silent = TRUE)
    debug_mode <- is.null(parent_desc)
    if (debug_mode) {
      parent_desc <- local_descriptor()
    }
    fwd_graph <- trace_fn(f, args_flat = prep$args_flat, in_tree = prep$in_tree)
    grad_graph <- transform_gradient(fwd_graph, wrt)
    # parent_desc is modified in place
    if (!debug_mode) {
      return(inline_graph_into_desc(parent_desc, grad_graph))
    }
    unflatten(grad_graph$out_tree, lapply(grad_graph$outputs, \(x) DebugBox(x$aval)))
  }
  formals(f_gradient) <- formals2(f)
  return(f_gradient)
}

#' @title Value and Gradient
#' @description
#' Returns a new function that computes both the output of `f` and its gradient in a
#' single forward+backward pass. The result is a named list with elements `value` (the
#' original return value of `f`) and `grad` (the gradients, structured like the inputs or
#' the `wrt` subset).
#' @inheritParams gradient
#' @return A function with the same formals as `f` that returns
#'   `list(value = ..., grad = ...)`.
#' @seealso [`gradient()`]
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' loss_fn <- function(x) sum(x^2L)
#' vg <- jit(value_and_gradient(loss_fn))
#' result <- vg(nv_tensor(c(3, 4), dtype = "f32"))
#' result$value
#' result$grad
value_and_gradient <- function(f, wrt = NULL) {
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
  f_value_and_grad <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    prep <- prepare_gradient_args(args, wrt)

    parent_desc <- .current_descriptor(silent = TRUE)
    debug_mode <- is.null(parent_desc)
    if (debug_mode) {
      parent_desc <- local_descriptor()
    }
    fwd_graph <- trace_fn(f, args_flat = prep$args_flat, in_tree = prep$in_tree)
    grad_graph <- transform_gradient(fwd_graph, wrt)

    combined_graph <- grad_graph
    combined_graph$outputs <- c(fwd_graph$outputs, grad_graph$outputs)

    counter <- new_counter()
    value_tree <- reindex_tree(fwd_graph$out_tree, counter)
    grad_tree <- reindex_tree(grad_graph$out_tree, counter)
    combined_graph$out_tree <- ListNode(
      list(value_tree, grad_tree),
      names = c("value", "grad")
    )
    if (!debug_mode) {
      return(inline_graph_into_desc(parent_desc, combined_graph))
    }
    unflatten(
      combined_graph$out_tree,
      c(
        lapply(fwd_graph$outputs, \(x) DebugBox(x$aval)),
        lapply(grad_graph$outputs, \(x) DebugBox(x$aval))
      )
    )
  }
  formals(f_value_and_grad) <- formals2(f)
  f_value_and_grad
}
