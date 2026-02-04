#' @include tensor.R
#' @include box.R

#' @title Graph Value
#' @description
#' Value in an [`AnvilGraph`]. This is a mutable class.
#' @param aval ([`AbstractTensor`])\cr
#'   The abstract value of the variable.
#' @return (`GraphValue`)
#' @export
GraphValue <- function(aval) {
  checkmate::assert_class(aval, "AbstractTensor")

  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  env$aval <- aval

  structure(env, class = "GraphValue")
}

#' @title Graph Literal
#' @description
#' Literal in an [`AnvilGraph`]. This is a mutable class.
#' @param aval ([`LiteralTensor`])\cr
#'   The value of the literal.
#' @return (`GraphLiteral`)
#' @export
GraphLiteral <- function(aval) {
  checkmate::assert_class(aval, "LiteralTensor")

  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  env$aval <- aval

  structure(env, class = "GraphLiteral")
}

is_graph_literal <- function(x) {
  inherits(x, "GraphLiteral")
}


#' @export
format.GraphValue <- function(x, ...) {
  sprintf("GraphValue(%s)", format(x$aval))
}

#' @export
print.GraphValue <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
format.GraphLiteral <- function(x, ...) {
  # otherwise there might be conversion issues, so we directly use the pjrt printer
  # instead of converting via as_array(), which loses precision
  val <- if (is_anvil_tensor(x$aval$data)) {
    trimws(capture.output(print(x$aval$data))[2L])
  } else {
    as.character(x$aval$data)
  }
  sprintf("GraphLiteral(%s, %s, %s)", val, dtype2string(x$aval$dtype, x$aval$ambiguous), shape2string(x$aval$shape))
}

#' @export
print.GraphLiteral <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @title Graph Node
#' @description
#' Virtual base class for nodes in an [`AnvilGraph`].
#' Is either a [`GraphValue`] or a [`GraphLiteral`].
#' Cannot be instantiated directly - use [`GraphValue()`] or [`GraphLiteral()`] instead.
#' @name GraphNode
NULL

#' @title Primitive Call
#' @description
#' Call of a primitive in an [`AnvilGraph`]
#' Note that a primitive call also be a call into another graph (`p_graph`).
#' @param primitive (`AnvilPrimitive`)\cr
#'   The function.
#' @param inputs (`list(GraphValue)`)\cr
#'   The (tensor) inputs to the primitive.
#' @param params (`list(<any>)`)\cr
#'   The (static) parameters of the function call.
#' @param outputs (`list(GraphValue)`)\cr
#'   The (tensor) outputs of the primitive.
#' @return (`PrimitiveCall`)
#' @export
PrimitiveCall <- function(primitive, inputs, params, outputs) {
  checkmate::assert_class(primitive, "AnvilPrimitive")
  checkmate::assert_list(inputs, types = c("GraphValue", "GraphLiteral"))
  checkmate::assert_list(params)
  checkmate::assert_list(outputs, c("GraphValue", "GraphLiteral"))

  structure(
    list(
      primitive = primitive,
      inputs = inputs,
      params = params,
      outputs = outputs
    ),
    class = "PrimitiveCall"
  )
}

#' @title Graph of Primitive Calls
#'
#' @description
#' Computational graph consisting exclusively of primitive calls.
#' This is a mutable class.
#'
#' @param calls (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#'   This can also be another call into a graph when the primitive is a `p_call`.
#' @param in_tree (`NULL | Node`)\cr
#'   The tree of inputs. May contain leaves for both tensor inputs and static
#'   (non-tensor) arguments. Only the tensor leaves correspond to entries in
#'   `inputs`; use `is_static_flat` to distinguish them.
#' @param out_tree (`NULL | Node`)\cr
#'   The tree of outputs.
#' @param inputs (`list(GraphValue)`)\cr
#'   The inputs to the graph (tensor arguments only).
#' @param outputs (`list(GraphValue)`)\cr
#'   The outputs of the graph.
#' @param constants (`list(GraphValue)`)\cr
#'   The constants of the graph.
#' @param is_static_flat (`NULL | logical()`)\cr
#'   Boolean mask indicating which flat positions in `in_tree` are static (non-tensor) args.
#'   `NULL` when all args are tensor inputs.
#' @return (`AnvilGraph`)
# @export
AnvilGraph <- function(
  calls = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  constants = list(),
  is_static_flat = NULL
) {
  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  env$calls <- calls
  env$in_tree <- in_tree
  env$out_tree <- out_tree
  env$inputs <- inputs
  env$outputs <- outputs
  env$constants <- constants
  env$is_static_flat <- is_static_flat

  structure(env, class = "AnvilGraph")
}

#' @title Graph Descriptor
#' @description
#' Descriptor of an [`AnvilGraph`]. This is a mutable class.
#' @param calls (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#' @param tensor_to_gval (`hashtab`)\cr
#'   Mapping: `AnvilTensor` -> `GraphValue`
#' @param gval_to_box (`hashtab`)\cr
#'   Mapping: `GraphValue` -> `GraphBox`
#' @param constants (`list(GraphValue)`)\cr
#'   The constants of the graph.
#' @param in_tree (`NULL | Node`)\cr
#'   The tree of inputs. May contain leaves for both tensor inputs and static
#'   (non-tensor) arguments. Only the tensor leaves correspond to entries in
#'   `inputs`; use `is_static_flat` to distinguish them.
#' @param out_tree (`NULL | Node`)\cr
#'   The tree of outputs.
#' @param inputs (`list(GraphValue)`)\cr
#'   The inputs to the graph (tensor arguments only).
#' @param outputs (`list(GraphValue)`)\cr
#'   The outputs of the graph.
#' @param is_static_flat (`NULL | logical()`)\cr
#'   Boolean mask indicating which flat positions in `in_tree` are static (non-tensor) args.
#'   `NULL` when all args are tensor inputs.
#' @return (`GraphDescriptor`)
#' @export
GraphDescriptor <- function(
  calls = list(),
  tensor_to_gval = NULL,
  gval_to_box = NULL,
  constants = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  is_static_flat = NULL
) {
  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  env$calls <- calls
  env$tensor_to_gval <- tensor_to_gval %||% hashtab()
  env$gval_to_box <- gval_to_box %||% hashtab()
  env$constants <- constants
  env$in_tree <- in_tree
  env$out_tree <- out_tree
  env$inputs <- inputs
  env$outputs <- outputs
  env$is_static_flat <- is_static_flat

  structure(env, class = "GraphDescriptor")
}

#' @export
shape.GraphValue <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.GraphValue <- function(x, ...) {
  dtype(x$aval)
}

#' @export
shape.GraphLiteral <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.GraphLiteral <- function(x, ...) {
  x$aval$dtype
}


is_graph_descriptor <- function(x) {
  inherits(x, "GraphDescriptor")
}

descriptor_to_graph <- function(descriptor) {
  graph <- AnvilGraph(
    calls = descriptor$calls,
    in_tree = descriptor$in_tree,
    out_tree = descriptor$out_tree,
    inputs = descriptor$inputs,
    outputs = descriptor$outputs,
    constants = descriptor$constants,
    is_static_flat = descriptor$is_static_flat
  )
  maybe_restore_previous_desc(descriptor)
  graph
}

# Now the graph-building

#' @title Graph Box
#' @description
#' Box that represents a node in a [`GraphDescriptor`].
#' @param gnode ([`GraphNode`])\cr
#'   The node.
#' @param desc ([`GraphDescriptor`])\cr
#'   The descriptor of the graph.
#' @return (`GraphBox`)
#' @export
GraphBox <- function(gnode, desc) {
  if (!is_graph_node(gnode)) {
    cli_abort("gnode must be a GraphValue or GraphLiteral")
  }
  checkmate::assert_class(desc, "GraphDescriptor")

  structure(
    list(gnode = gnode, desc = desc),
    class = c("GraphBox", "AnvilBox")
  )
}

#' @export
shape.GraphBox <- function(x, ...) {
  shape(x$gnode)
}

#' @export
dtype.GraphBox <- function(x, ...) {
  dtype(x$gnode)
}

#' @export
print.GraphBox <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
format.GraphBox <- function(x, ...) {
  sprintf("GraphBox(%s)", format(x$gnode))
}

maybe_box_tensorish <- function(x) {
  current_desc <- .current_descriptor()
  if (is_graph_box(x)) {
    if (identical(x$desc, current_desc)) {
      return(x)
    }
    gval <- x$gnode
    get_box_or_register_const(current_desc, gval)
  } else if (is_anvil_tensor(x) || is_lit(x)) {
    get_box_or_register_const(current_desc, x)
  } else if (is_debug_box(x)) {
    # We want debug mode to emulate standard tracing, so each primitive initializes it's own
    # GraphDescriptor during debug mode and we evaluate with GraphBox objects
    # before returning to the user, the GraphBox is converted to a DebugBox again
    GraphBox(GraphValue(aval = x$aval), current_desc)
  } else if (is_abstract_tensor(x)) {
    cli_abort("Don't use AbtractTensors as inputs; For debugging, use `debug_box()`")
  } else {
    cli_abort("Expected tensorish value, but got {.cls {class(x)[1]}}")
  }
}

# this function is on the inputs of trace_fn()
maybe_box_input <- function(x, desc, toplevel, lit_to_tensor) {
  if (lit_to_tensor && test_scalar(x)) {
    # so we can accept literals as inputs to higher-order primitives like if and while
    ambiguous <- !is.logical(x)
    gval <- GraphValue(
      aval = AbstractTensor(
        dtype = default_dtype(x),
        shape = integer(),
        ambiguous = ambiguous
      )
    )
    return(register_input(desc, gval))
  }
  if (is_anvil_tensor(x)) {
    # cases:
    # 1. top-level trace_fn call
    # 2. a constant is passed to a nested trace_fn call
    #    this constant can be a closed-over constant or defined in the environment of the nested trace_fn call
    # For the first scenario, it would be sufficient to create a AbstractTensor,
    # because the input will be provided by the user
    # For the second scenario, we will inline the descriptor into the parent descriptor,
    # but it the input to the nested trace_fn call does not become an input to the parent graph,
    # but is simply an existing value, that is a value from the parent graph
    # however, if the value does not exist in the parent graph, we need to add it as a constant
    # for that, we need to keep the value of the actual tensor, so we can later register it
    # see test: "can pass constant to nested trace_fn call if it ..." in test-graph.R
    gval <- if (toplevel) {
      # user-provided inputs are simply unknown
      GraphValue(aval = to_abstract(x, pure = TRUE))
    } else {
      # nested trace_fn call might receive known constants from the parent graph as input
      GraphValue(aval = ConcreteTensor(x))
    }
    register_input(desc, gval)
  } else if (is_debug_box(x)) {
    # User provided abstract input
    # This is useful for debugging and in jit() we anyway verify that the inputs are AnvilTensors
    # so we don't accidentally box abstract tensors there
    gval <- GraphValue(aval = x$aval)
    register_input(desc, gval)
  } else if (is_graph_box(x)) {
    # Nested trace_fn call
    # Because we will inline the child graph into the parent graph, we re-use
    # the same GraphValue, because this will make the inlining straightforward.
    register_input(desc, x$gnode)
  } else if (is_abstract_tensor(x)) {
    # Needed to be able to pass abstract tensors to trace_fn()
    gval <- GraphValue(aval = x)
    register_input(desc, gval)
  } else {
    if (lit_to_tensor) {
      cli_abort("Expected only tensorish values, but got {.cls {class(x)[1]}}")
    }
    # parameter
    x
  }
}

register_input <- function(desc, x) {
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register an input in a non-graph descriptor")
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid input")
  }
  desc$inputs <- c(desc$inputs, list(x))
  box <- GraphBox(x, desc)
  desc$gval_to_box[[x]] <- box
  box
}

register_gval <- function(desc, x) {
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register a gval in a non-graph descriptor")
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid gval")
  }
  box <- desc$gval_to_box[[x]]
  if (!is.null(box)) {
    return(box)
  }
  box <- GraphBox(x, desc)
  desc$gval_to_box[[x]] <- box
  box
}

# Returns a Box
get_box_or_register_const <- function(desc, x) {
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register a constant in a non-graph descriptor")
  }
  if (is_anvil_tensor(x)) {
    gval <- desc$tensor_to_gval[[x]]
    if (!is.null(gval)) {
      return(desc$gval_to_box[[gval]])
    }
    gval <- GraphValue(aval = ConcreteTensor(x))
    desc$tensor_to_gval[[x]] <- gval
    desc$constants <- c(desc$constants, list(gval))
    box <- GraphBox(gval, desc)
    desc$gval_to_box[[gval]] <- box
    return(box)
  }
  if (test_scalar(x)) {
    ambiguous <- !is.logical(x)
    gval <- GraphLiteral(LiteralTensor(x, shape = integer(), ambiguous = ambiguous))
    box <- desc$gval_to_box[[gval]] <- GraphBox(gval, desc)
    return(box)
  }
  if (is_graph_literal(x)) {
    box <- desc$gval_to_box[[x]] <- GraphBox(x, desc)
    return(box)
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid constant")
  }
  # gval$aval can either be a
  # * ConcreteTensor: AnvilTensor that is captured from the parent environment
  # * AbstractTensor: Output of a computation in a parent graph
  # In either case, we first check whether the value is already registered in the current graph
  # and if so, return it:
  box <- desc$gval_to_box[[x]]
  if (!is.null(box)) {
    return(box)
  }

  # Now, we create the new box and register it, so if we see it again, we can return it immediately.
  new_box <- GraphBox(x, desc)

  if (is_concrete_tensor(x$aval)) {
    desc$tensor_to_gval[[x$aval$data]] <- x
  }
  desc$gval_to_box[[x]] <- new_box
  desc$constants <- c(desc$constants, list(x))
  return(new_box)
}

init_desc_from_graph <- function(desc, graph, outputs = TRUE) {
  for (input in graph$inputs) {
    register_input(desc, input)
  }
  for (const in graph$constants) {
    get_box_or_register_const(desc, const)
  }
  for (call in graph$calls) {
    for (input in c(call$inputs, call$outputs)) {
      if (is.null(desc$gval_to_box[[input]])) {
        desc$gval_to_box[[input]] <- GraphBox(input, desc)
      }
    }
  }

  desc$calls <- graph$calls
  desc$in_tree <- graph$in_tree
  if (outputs) {
    desc$outputs <- graph$outputs
  }
  desc$out_tree <- graph$out_tree

  graph
}

#' @title Trace an R function into a Graph
#' @description
#' Create a graph representation of an R function by tracing.
#' @param f (`function`)\cr
#'   The function to trace_fn.
#' @param args (`list` of ([`AnvilTensor`] | [`AbstractTensor`]))\cr
#'   The (unflattened) arguments to the function.
#' @param desc (`NULL` | `GraphDescriptor`)\cr
#'   The descriptor to use for the graph.
#' @param toplevel (`logical(1)`)\cr
#'   Whether the function is being traced at the top level.
#'   If this is `TRUE`, inputs that are `AnvilTensor`s are treated as unknown.
#'   If this is `FALSE` (default), `AnvilTensor`s are treated as constants.
#' @param lit_to_tensor (`logical(1)`)\cr
#'   Whether to convert literal inputs to `AnvilTensor`s.
#'   Should only be used for higher-order primitives like if and while, where no static inputs are possible.
#' @param args_flat (`list`)\cr
#'   The flattened arguments. Also requires passing `in_tree`.
#' @param in_tree (`Node`)\cr
#'   The tree structure of the arguments.
#' @return ([`AnvilGraph`])
#' @export
trace_fn <- function(
  f,
  args = NULL,
  desc = NULL,
  toplevel = FALSE,
  lit_to_tensor = FALSE,
  args_flat = NULL,
  in_tree = NULL
) {
  if (inherits(f, "JitFunction")) {
    cli_abort("{.arg f} must not be a jitted function.")
  }
  if (is.null(args)) {
    if (is.null(args_flat) || is.null(in_tree)) {
      cli_abort("args or args_flat and in_tree must be provided")
    }
  } else {
    if (!is.null(args_flat) || !is.null(in_tree)) {
      cli_abort("args and args_flat and in_tree must not be provided together")
    }
    in_tree <- build_tree(args)
    args_flat <- flatten(args)
  }
  f_flat <- flatten_fun(f, in_node = in_tree)
  if (is.null(desc)) {
    desc <- local_descriptor(in_tree = in_tree)
  } else {
    desc$in_tree <- in_tree
  }

  # box tensors and add them as inputs to the current graph
  inputs_flat <- lapply(args_flat, maybe_box_input, desc = desc, toplevel = toplevel, lit_to_tensor = lit_to_tensor)
  # Track which flat args are static (non-tensor) values vs. graph inputs
  desc$is_static_flat <- vapply(inputs_flat, Negate(is_graph_box), logical(1L))
  output <- do.call(f_flat, inputs_flat)

  out_tree <- output[[1L]]
  # function() x; -> output can be an closed-over constant
  outputs_flat <- lapply(output[[2L]], maybe_box_tensorish)

  desc$out_tree <- out_tree
  desc$outputs <- lapply(outputs_flat, \(x) x$gnode)

  if (any(vapply(outputs_flat, \(x) !is_graph_box(x), logical(1L)))) {
    cli_abort("Function .f must return only objects of type `GraphBox`.")
  }

  graph <- descriptor_to_graph(desc)
  return(graph)
}

is_graph_node <- function(x) {
  is_graph_value(x) || is_graph_literal(x)
}

is_graph_value <- function(x) {
  inherits(x, "GraphValue")
}

maybe_restore_previous_desc <- function(desc = NULL) {
  if (!is.null(desc) && (!identical(desc, globals[["CURRENT_DESCRIPTOR"]]))) {
    # graph has already been returned
    return()
  }

  stash_size <- length(globals[["DESCRIPTOR_STASH"]])
  if (stash_size) {
    globals[["CURRENT_DESCRIPTOR"]] <- globals[["DESCRIPTOR_STASH"]][[stash_size]]
    globals[["DESCRIPTOR_STASH"]] <- globals[["DESCRIPTOR_STASH"]][-stash_size]
  } else {
    globals[["CURRENT_DESCRIPTOR"]] <- NULL
  }
}

#' @title Get the current graph
#' @description
#' Get the current graph being built (via [`local_descriptor`]).
#' @param silent (`logical(1)`)\cr
#'   Whether to return `NULL` if no graph is currently being built (as opposed to aborting).
#' @return A [`GraphDescriptor`] object.
#' @export
.current_descriptor <- function(silent = FALSE) {
  maybe_desc <- globals[["CURRENT_DESCRIPTOR"]]
  if (silent) {
    return(maybe_desc)
  }
  maybe_desc %??%
    cli_abort("No graph is currently being built. Did you forget to use `jit()`?")
}

#' @title Create a graph
#' @description
#' Creates a new [`GraphDescriptor`] which is afterwards accessible via [`.current_descriptor()`].
#' The graph is automatically removed when exiting the current scope.
#' After the graph is either cleaned up automatically (by exiting the scope)
#' or finalized, the previously built graph is restored,
#' i.e., accessible via [`.current_descriptor()`].
#'
#' @param envir (`environment`)\cr
#'   Environment where exit handler will be registered for cleaning up the
#'   [`GraphDescriptor`] if it was not returned yet.
#' @param ... (`any`)\cr
#'   Additional arguments to pass to the [`GraphDescriptor`] constructor.
#' @return A [`GraphDescriptor`] object.
#' @export
local_descriptor <- function(..., envir = parent.frame()) {
  if (identical(envir, globalenv())) {
    # lingering global descriptors mess with our debug mode
    cli_abort("Don't run local_descriptor in the global environment")
  }

  desc <- GraphDescriptor(...)
  if (!is.null(globals[["CURRENT_DESCRIPTOR"]])) {
    globals[["DESCRIPTOR_STASH"]] <- c(
      globals[["DESCRIPTOR_STASH"]],
      list(globals[["CURRENT_DESCRIPTOR"]])
    )
  }
  globals[["CURRENT_DESCRIPTOR"]] <- desc

  withr::defer(
    envir = envir,
    {
      maybe_restore_previous_desc(desc)
    },
    priority = "first"
  )
  return(desc)
}

is_graph <- function(x) {
  inherits(x, "AnvilGraph")
}
is_graph_box <- function(x) {
  inherits(x, "GraphBox")
}

#' @title Add a Primitive Call to a Graph Descriptor
#' @description
#' Add a primitive call to a graph descriptor.
#' @param prim ([`AnvilPrimitive`])\cr
#'   The primitive to add.
#' @param args (`list` of [`GraphNode`])\cr
#'   The arguments to the primitive.
#' @param params (`list`)\cr
#'   The parameters to the primitive.
#' @param infer_fn (`function`)\cr
#'   The inference function to use.
#'   Must output a list of [`AbstractTensor`]s.
#' @param desc ([`GraphDescriptor`] | `NULL`)\cr
#'   The graph descriptor to add the primitive call to.
#'   Uses the [current descriptor][.current_descriptor] if `NULL`.
#' @param debug_mode (`logical(1)`)\cr
#'   Whether to just perform abstract evaluation for debugging.
#' @return (`list` of `Box`)\cr
#'   Either `GraphBox` objects or `DebugBox` objects, depending on `debug_mode`.
#' @export
graph_desc_add <- function(prim, args, params = list(), infer_fn, desc = NULL, debug_mode = NULL) {
  desc <- desc %??% .current_descriptor(silent = TRUE)

  debug_mode <- debug_mode %??% is.null(desc)
  if (debug_mode && is.null(desc)) {
    desc <- local_descriptor()
  }

  boxes_in <- lapply(args, maybe_box_tensorish)
  gnodes_in <- unname(lapply(boxes_in, \(box) box$gnode))
  avals_in <- lapply(boxes_in, \(box) box$gnode$aval)
  ats_out <- tryCatch(
    {
      rlang::exec(infer_fn, !!!c(avals_in, params))
    },
    error = function(e) {
      e$call <- print_call_repr(prim)
      e <- stablehlo::to_one_based(e)
      rlang::cnd_signal(e)
    }
  )
  gvals_out <- lapply(ats_out, GraphValue)
  call <- PrimitiveCall(prim, gnodes_in, params, gvals_out)
  desc$calls <- c(desc$calls, list(call))
  boxes_out <- lapply(gvals_out, register_gval, desc = desc)
  if (debug_mode) {
    return(lapply(boxes_out, \(x) DebugBox(to_abstract(x))))
  }
  return(boxes_out)
}

print_call_repr <- function(prim) {
  rlang::exec(call, paste0("nvl_", prim$name))
}


inline_graph_into_desc <- function(desc, graph) {
  for (const in graph$constants) {
    # The following can happen:
    # 1. a constant is already present in the parent descriptor -> do nothing
    # 2. the constant is not present in the parent descriptor -> register it
    get_box_or_register_const(desc, const)
  }
  for (input in graph$inputs) {
    if (is.null(desc$gval_to_box[[input]])) {
      #
    }
    get_box_or_register_const(desc, input)
  }

  desc$calls <- c(desc$calls, graph$calls)

  gvals_out_flat <- graph$outputs
  boxes_out_flat <- lapply(gvals_out_flat, GraphBox, desc)
  unflatten(graph$out_tree, boxes_out_flat)
}
