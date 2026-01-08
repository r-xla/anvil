#' @include tensor.R
#' @include box.R

#' @title Graph Value
#' @description
#' Value in a [`Graph`].
#' @param aval ([`AbstractTensor`])\cr
#'   The abstract value of the variable.
#' @include mut.R
GraphValue <- mut(new_class(
  "GraphValue",
  properties = list(
    aval = AbstractTensor
  )
))

#' @title Graph Literal
#' @description
#' Literal in a [`Graph`].
#' @param aval (`any`)\cr
#'   The value of the literal.
#' @export
GraphLiteral <- mut(new_class(
  "GraphLiteral",
  properties = list(
    aval = LiteralTensor
  )
))

is_graph_literal <- function(x) {
  inherits(x, "anvil::mut<GraphLiteral>")
}

method(format, GraphValue) <- function(x, ...) {
  sprintf("GraphValue(%s)", format(x@aval))
}

method(format, GraphLiteral) <- function(x, ...) {
  sprintf("GraphLiteral(%s, %s)", x@aval@data, sprintf("%s%s", repr(x@aval@dtype), if (x@aval@ambiguous) "?" else ""))
}

method(print, GraphLiteral) <- function(x, ...) {
  cat(format(x), "\n")
}

#' @title Graph Node
#' @description
#' Node in a [`Graph`].
#' Is either a [`GraphValue`] or a [`GraphLiteral`].
#' @export
GraphNode <- S7::new_union(GraphValue, GraphLiteral)
# TODO(rename): It's actually more like an edge ...

#' @title Primitive Call
#' @description
#' Call of a primitive in a [`Graph`]
#' Note that a primitive call also be a call into another graph (`p_graph`).
#' @param primitive (`Primitive`)\cr
#'   The function.
#' @param inputs (`list(GraphValue)`)\cr
#'   The (tensor) inputs to the primitive.
#' @param params (`list(<any>)`)\cr
#'   The (static) parameters of the function call.
#' @param outputs (`list(GraphValue)`)\cr
#'   The (tensor) outputs of the primitive.
#' @export
PrimitiveCall <- new_class(
  "PrimitiveCall",
  properties = list(
    primitive = Primitive,
    inputs = list_of(GraphNode),
    params = list_of(class_any),
    outputs = list_of(GraphNode)
  )
)

#' @title Graph of Primitive Calls
#'
#' @description
#' Computational graph consisting exclusively of primitive calls.
#'
#' @param calls (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#'   This can also be another call into a graph when the primitive is a `p_call`.
#' @param in_tree (`NULL | Node`)\cr
#'   The tree of inputs.
#' @param out_tree (`NULL | Node`)\cr
#'   The tree of outputs.
#' @param inputs (`list(GraphValue)`)\cr
#'   The inputs to the graph.
#' @param outputs (`list(GraphValue)`)\cr
#'   The outputs of the graph.
#' @param constants (`list(GraphValue)`)\cr
#'   The constants of the graph.
# @export
Graph <- mut(new_class(
  "Graph",
  properties = list(
    # Primitive: list(GraphValue) --[params]--> list(GraphValue)
    # All the GraphValues that are inputs will already have a binding.
    # Those that are constants as well
    calls = list_of(PrimitiveCall),
    ## Used to (un-)flatten inputs and outputs
    in_tree = NULL | new_S3_class("Node"),
    out_tree = NULL | new_S3_class("Node"),
    inputs = list_of(GraphValue),
    outputs = list_of(GraphNode),
    constants = list_of(GraphValue)
  )
))

#' @title Graph Descriptor
#' @description
#' Descriptor of a [`Graph`].
#' @param calls (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#' @param tensor_to_gval (`hashtab`)\cr
#'   Mapping: `AnvilTensor` -> `GraphValue`
#' @param gval_to_box (`hashtab`)\cr
#'   Mapping: `GraphValue` -> `GraphBox`
#' @param constants (`list(GraphValue)`)\cr
#'   The constants of the graph.
#' @param in_tree (`NULL | Node`)\cr
#'   The tree of inputs.
#' @param out_tree (`NULL | Node`)\cr
#'   The tree of outputs.
#' @param inputs (`list(GraphValue)`)\cr
#'   The inputs to the graph.
#' @param outputs (`list(GraphValue)`)\cr
#'   The outputs of the graph.
#' @export
GraphDescriptor <- mut(new_class(
  "GraphDescriptor",
  properties = list(
    calls = list_of(PrimitiveCall),
    tensor_to_gval = new_property(class_hashtab, default = quote(hashtab())),
    gval_to_box = new_property(class_hashtab, default = quote(hashtab())),
    constants = list_of(GraphValue),
    in_tree = NULL | new_S3_class("Node"),
    out_tree = NULL | new_S3_class("Node"),
    inputs = list_of(GraphValue),
    outputs = list_of(GraphNode)
  )
))

method(shape, GraphValue) <- function(x, ...) {
  shape(x@aval)
}

method(dtype, GraphValue) <- function(x, ...) {
  dtype(x@aval)
}

method(shape, GraphLiteral) <- function(x, ...) {
  shape(x@aval)
}

method(dtype, GraphLiteral) <- function(x, ...) {
  x@aval@dtype
}


is_graph_descriptor <- function(x) {
  inherits(x, "anvil::mut<GraphDescriptor>")
}

descriptor_to_graph <- function(descriptor) {
  graph <- Graph(
    calls = descriptor@calls,
    inputs = descriptor@inputs,
    outputs = descriptor@outputs,
    constants = descriptor@constants
  )
  graph@in_tree <- descriptor@in_tree
  graph@out_tree <- descriptor@out_tree
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
#' @export
GraphBox <- new_class(
  "GraphBox",
  parent = Box,
  properties = list(
    gnode = GraphNode,
    desc = GraphDescriptor
  )
)

method(shape, GraphBox) <- function(x, ...) {
  shape(x@gnode)
}

method(dtype, GraphBox) <- function(x, ...) {
  dtype(x@gnode)
}

method(print, GraphBox) <- function(x, ...) {
  cat(format(x), "\n")
}

method(format, GraphBox) <- function(x, ...) {
  sprintf("GraphBox(%s)", format(x@gnode))
}

maybe_box_variable <- function(x) {
  current_desc <- .current_descriptor()
  if (is_graph_box(x)) {
    if (identical(x@desc, current_desc)) {
      return(x)
    }
    gval <- x@gnode
    get_box_or_register_const(current_desc, gval)
  } else if (is_anvil_tensor(x) || test_scalar(x)) {
    get_box_or_register_const(current_desc, x)
  } else if (is_graph_node(x)) {
    # FIXME: !!!
    # We use this in gradient, where we pass gvals to the backward rules
    # but I think we should handle this differently
    GraphBox(x, current_desc)
  } else if (is_debug_box(x)) {
    # We want debug mode to emulate standard tracing, so each primitive initializes it's own
    # GraphDescriptor during debug mode and we evaluate with GraphBox objects
    # before returning to the user, the GraphBox is converted to a DebugBox again
    GraphBox(GraphValue(aval = x@aval), current_desc)
  } else if (is_abstract_tensor(x)) {
    cli_abort("Don't use AbtractTensors as inputs; For debugging, use `debug_box()`")
  } else {
    x
  }
}

# this function is on the inputs of trace_fn()
maybe_box_input <- function(x, desc, toplevel) {
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
    gval <- GraphValue(aval = x@aval)
    register_input(desc, gval)
  } else if (is_graph_box(x)) {
    # Nested trace_fn call
    # Because we will inline the child graph into the parent graph, we re-use
    # the same GraphValue, because this will make the inlining straightforward.
    register_input(desc, x@gnode)
  } else if (is_abstract_tensor(x)) {
    # Needed to be able to pass abstract tensors to trace_fn()
    gval <- GraphValue(aval = x)
    register_input(desc, gval)
  } else {
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
  desc@inputs <- c(desc@inputs, x)
  box <- GraphBox(x, desc)
  desc@gval_to_box[[x]] <- box
  box
}

register_gval <- function(desc, x) {
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register a gval in a non-graph descriptor")
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid gval")
  }
  box <- desc@gval_to_box[[x]]
  if (!is.null(box)) {
    return(box)
  }
  box <- GraphBox(x, desc)
  desc@gval_to_box[[x]] <- box
  box
}

# Returns a Box
get_box_or_register_const <- function(desc, x) {
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register a constant in a non-graph descriptor")
  }
  if (is_anvil_tensor(x)) {
    gval <- desc@tensor_to_gval[[x]]
    if (!is.null(gval)) {
      return(desc@gval_to_box[[gval]])
    }
    gval <- GraphValue(aval = ConcreteTensor(x))
    desc@tensor_to_gval[[x]] <- gval
    desc@constants <- c(desc@constants, gval)
    box <- GraphBox(gval, desc)
    desc@gval_to_box[[gval]] <- box
    return(box)
  }
  if (test_scalar(x)) {
    ambiguous <- !is.logical(x)
    gval <- GraphLiteral(LiteralTensor(x, shape = integer(), ambiguous = ambiguous))
    box <- desc@gval_to_box[[gval]] <- GraphBox(gval, desc)
    return(box)
  }
  if (is_graph_literal(x)) {
    box <- desc@gval_to_box[[x]] <- GraphBox(x, desc)
    return(box)
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid constant")
  }
  # gval@aval can either be a
  # * ConcreteTensor: AnvilTensor that is captured from the parent environment
  # * AbstractTensor: Output of a computation in a parent graph
  # In either case, we first check whether the value is already registered in the current graph
  # and if so, return it:
  box <- desc@gval_to_box[[x]]
  if (!is.null(box)) {
    return(box)
  }

  # Now, we create the new box and register it, so if we see it again, we can return it immediately.
  new_box <- GraphBox(x, desc)

  if (is_concrete_tensor(x@aval)) {
    desc@tensor_to_gval[[x@aval@data]] <- x
  }
  desc@gval_to_box[[x]] <- new_box
  desc@constants <- c(desc@constants, x)
  return(new_box)
}

init_desc_from_graph <- function(desc, graph, outputs = TRUE) {
  for (input in graph@inputs) {
    register_input(desc, input)
  }
  for (const in graph@constants) {
    get_box_or_register_const(desc, const)
  }
  for (call in graph@calls) {
    for (input in c(call@inputs, call@outputs)) {
      if (is.null(desc@gval_to_box[[input]])) {
        desc@gval_to_box[[input]] <- GraphBox(input, desc)
      }
    }
  }

  desc@calls <- graph@calls
  desc@in_tree <- graph@in_tree
  if (outputs) {
    desc@outputs <- graph@outputs
  }
  desc@out_tree <- graph@out_tree

  graph
}

#' @title Trace an R function into a Graph
#' @description
#' Create a graph representation of an R function by tracing.
#' @param f (`function`)\cr
#'   The function to trace_fn.
#' @param args (`list` of ([`AnvilTensor`] | [`AbstractTensor`]))\cr
#'   The arguments to the function.
#' @param desc (`NULL` | `GraphDescriptor`)\cr
#'   The descriptor to use for the graph.
#' @param toplevel (`logical(1)`)\cr
#'   Whether the function is being traced at the top level.
#'   If this is `TRUE`, inputs that are `AnvilTensor`s are treated as unknown.
#'   If this is `FALSE` (default), `AnvilTensor`s are treated as constants.
#' @return ([`Graph`])
#' @export
trace_fn <- function(f, args, desc = NULL, toplevel = FALSE) {
  in_tree <- build_tree(args)
  args_flat <- flatten(args)
  f_flat <- flatten_fun(f, in_node = in_tree)
  if (is.null(desc)) {
    desc <- local_descriptor(in_tree = in_tree)
  } else {
    desc@in_tree <- in_tree
  }

  # box tensors and add them as inputs to the current graph
  inputs_flat <- lapply(args_flat, maybe_box_input, desc = desc, toplevel = toplevel)
  output <- do.call(f_flat, inputs_flat)

  out_tree <- output[[1L]]
  # function() x; -> output can be an closed-over constant
  outputs_flat <- lapply(output[[2L]], maybe_box_variable)

  desc@out_tree <- out_tree
  desc@outputs <- lapply(outputs_flat, \(x) x@gnode)

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
  inherits(x, "anvil::mut<GraphValue>")
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
#' Creates a new [`Graph`] which is afterwards accessible via [`.current_descriptor()`].
#' The graph is automatically removed when exiting the current scope.
#' After the graph is either cleaned up automatically (by exiting the scope)
#' or finalized, the previously built graph is restored,
#' i.e., accessible via [`.current_descriptor()`].
#'
#' @param envir (`environment`)\cr
#'   Environment where exit handler will be registered for cleaning up the
#'   [`Graph`] if it was not returned yet.
#' @param ... (`any`)\cr
#'   Additional arguments to pass to the [`GraphDescriptor`] constructor.
#' @return A [`Graph`] object.
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
  inherits(x, "anvil::mut<Graph>")
}
is_graph_box <- function(x) {
  inherits(x, "anvil::GraphBox")
}

#' @title Add a Primitive Call to a Graph Descriptor
#' @description
#' Add a primitive call to a graph descriptor.
#' @param prim ([`Primitive`])\cr
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

  boxes_in <- lapply(args, maybe_box_variable)
  gnodes_in <- lapply(boxes_in, \(box) box@gnode)
  avals_in <- lapply(boxes_in, \(box) box@gnode@aval)
  sts_out <- rlang::exec(infer_fn, !!!c(avals_in, params))
  gvals_out <- lapply(sts_out, GraphValue)
  call <- PrimitiveCall(prim, gnodes_in, params, gvals_out)
  desc@calls <- c(desc@calls, call)
  boxes_out <- lapply(gvals_out, register_gval, desc = desc)
  if (debug_mode) {
    return(lapply(boxes_out, \(x) DebugBox(to_abstract(x))))
  }
  return(boxes_out)
}


inline_graph_into_desc <- function(desc, graph) {
  for (const in graph@constants) {
    # The following can happen:
    # 1. a constant is already present in the parent descriptor -> do nothing
    # 2. the constant is not present in the parent descriptor -> register it
    get_box_or_register_const(desc, const)
  }
  for (input in graph@inputs) {
    if (is.null(desc@gval_to_box[[input]])) {
      #
    }
    get_box_or_register_const(desc, input)
  }

  desc@calls <- c(desc@calls, graph@calls)

  gvals_out_flat <- graph@outputs
  boxes_out_flat <- lapply(gvals_out_flat, GraphBox, desc)
  unflatten(graph@out_tree, boxes_out_flat)
}
