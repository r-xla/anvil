#' @title Graph Value
#' @description
#' Value in a [`Graph`].
#' @section Fields:
#' * `state` :: (`any`)\cr
#'   The state of the variable. Populated when the graph is executed.
#' @include mut.R
GraphValue <- mut(new_class(
  "GraphValue",
  properties = list(
    aval = ShapedTensor
  )
))

#' @title Graph Literal
#' @description
#' Literal in a [`Graph`].
#' @section Fields:
#' * `aval` :: (`any`)\cr
#'   The value of the literal.
#' * `dtype` :: (`stablehlo::TensorDataType`)\cr
#'   The dtype of the literal.
#' @export
GraphLiteral <- mut(new_class(
  "GraphLiteral",
  properties = list(
    aval = new_property(class_any, validator = function(value) {
      checkmate::check_scalar(value)
    }),
    dtype = stablehlo::TensorDataType
  )
))

is_graph_literal <- function(x) {
  inherits(x, "anvil::mut<GraphLiteral>")
}

method(format, GraphValue) <- function(x, ...) {
  sprintf("GraphValue(%s)", format(x@aval))
}

method(format, GraphLiteral) <- function(x, ...) {
  sprintf("GraphLiteral(%s, %s)", format(x@aval), format(x@dtype))
}

GraphNode <- S7::new_union(GraphValue, GraphLiteral)

#' @title Primitive Call
#' @description
#' Call of a primitive in a [`Graph`]
#' Note that a primitive call also be a call into another graph (`p_graph`).
#' @section Fields:
#' * `primitive` :: ([`Primitive`])\cr
#'   The function.
#' * `inputs` :: (`list(GraphValue)`)\cr
#'   The (tensor) inputs to the primitive.
#' * `params` :: (`list(<any>)`)\cr
#'   The (static) parameters of the function call.
#' * `outputs` :: (`list(GraphValue)`)\cr
#'   The (tensor) outputs of the primitive.
#' @export
PrimitiveCall <- new_class(
  "PrimitiveCall",
  properties = list(
    primitive = Primitive,
    params = list_of(class_any),
    inputs = list_of(GraphNode),
    outputs = list_of(GraphNode)
  )
)

#' @title Graph of Primitive Calls
#'
#' @description
#' Computational graph consisting exclusively of primitive calls.
#'
#' @section Fields:
#' * `calls` :: (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#'   This can also be another call into a graph when the primitive is a `p_call`.
#' * `in_tree` :: (`NULL | Node`)\cr
#'   The tree of inputs.
#' * `out_tree` :: (`NULL | Node`)\cr
#'   The tree of outputs.
#' * `inputs` :: (`list(GraphValue)`)\cr
#'   The inputs to the graph.
#' * `outputs` :: (`list(GraphValue)`)\cr
#'   The outputs of the graph.
#'
#' @export
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
#' @section Fields:
#' * `calls` :: (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#' * `tensor_to_gval` :: (`hashtab`)\cr
#'   Mapping: `AnvilTensor` -> `GraphValue`
#' * `gval_to_box` :: (`hashtab`)\cr
#'
#' @details
#' The trickiest thing in our setup are how we ensure that the same values receive the same identifier
#' (GraphValue) across nested graphs.
#' There are two cases:
#' 1. When a
GraphDescriptor <- mut(new_class(
  "GraphDescriptor",
  properties = list(
    calls = list_of(PrimitiveCall),
    # We either get boxes as GraphValue or Tensor
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
  integer()
}

method(dtype, GraphLiteral) <- function(x, ...) {
  x@dtype
}

# identical() fails for some reason on graph descriptors with the same .state
method(`==`, list(GraphDescriptor, GraphDescriptor)) <- function(e1, e2) {
  identical(e1@.state, e2@.state)
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

GraphBox <- new_class(
  "GraphBox",
  parent = Box,
  properties = list(
    # TODO: rename to gnode
    gval = GraphNode,
    desc = GraphDescriptor
  )
)

method(shape, GraphBox) <- function(x, ...) {
  shape(x@gval)
}

method(dtype, GraphBox) <- function(x, ...) {
  dtype(x@gval)
}

method(print, GraphBox) <- function(x, ...) {
  cat(format(x), "\n")
}

method(format, GraphBox) <- function(x, ...) {
  sprintf("GraphBox(%s)", format(x@gval))
}

aval <- function(x) {
  if (is_anvil_tensor(x)) {
    return(ConcreteTensor(x))
  }
  if (is_graph_box(x)) {
    return(x@gval@aval)
  }
  cli_abort("internal error")
}



maybe_box_variable <- function(x) {
  current_desc <- .current_descriptor()
  if (is_graph_box(x)) {
    if (x@desc == current_desc) {
      return(x)
    }
    gval <- x@gval
    get_box_or_register_cont(current_desc, gval)
  } else if (is_anvil_tensor(x)) {
    get_box_or_register_cont(current_desc, x)
  } else if (is_graph_node(x)) {
    # FIXME: !!!
    # We use this in gradient, but I am not sure this is such a great idea
    #browser()
    #cli_abort("Internal error: trying to lift a GraphNode")
    get_box_or_register_cont(current_desc, x)
    #GraphBox(x, .current_descriptor())
  } else {
    x
  }
}

maybe_box_input <- function(x, desc) {
  # this function is on the inputs of graphify()
  if (is_anvil_tensor(x)) {
    # cases:
    # 1. top-level graphify call
    # 2. a constant is passed to a nested graphify call
    #    this constant can be a closed-over constant or defined in the environment of the nested graphify call
    # For the first scenario, it would be sufficient to create a ShapedTensor,
    # because the input will be provided by the user
    # For the second scenario, we will inline the descriptor into the parent descriptor,
    # but it the input to the nested graphify call does not become an input to the parent graph,
    # but is simply an existing value, that is a value from the parent graph
    # however, if the value does not exist in the parent graph, we need to add it as a constant
    # for that, we need to keep the value of the actual tensor, so we can later register it
    # see test: "can pass constant to nested graphify call if it ..." in test-graph.R
    gval <- GraphValue(aval = ConcreteTensor(x))
    register_input(desc, gval)
  } else if (is_graph_box(x)) {
    # Nested graphify call
    # Because we will inline the child graph into the parent graph, we re-use
    # the same GraphValue, because this will make the inlining straightforward.
    register_input(desc, x@gval)
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
get_box_or_register_cont <- function(desc, x) {
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
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid constant")
  }
  # gval@aval can either be a
  # * ConcreteTensor: AnvilTensor that is captured from the parent environment
  # * ShapedTensor: Output of a computation in a parent graph
  # In either case, we first check whether the value is already registered in the current graph
  # and if so, return it:
  box <- desc@gval_to_box[[x]]
  if (!is.null(box)) {
    return(box)
  }

  # Now, we create the new box and register it, so if we see it again, we can return it immediately.
  new_box <- GraphBox(x, desc)

  if (is_concrete_tensor(aval)) {
    desc@tensor_to_gval[[x@data]] <- x
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
    get_box_or_register_cont(desc, const)
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

graphify <- function(f, args) {
  in_tree <- build_tree(args)
  args_flat <- flatten(args)
  f_flat <- flatten_fun(f, in_node = in_tree)
  desc <- local_descriptor(in_tree = in_tree)

  # box tensors and add them as inputs to the current graph
  inputs_flat <- lapply(args_flat, maybe_box_input, desc = desc)
  output <- do.call(f_flat, inputs_flat)

  out_tree <- output[[1L]]
  # function() x; -> output can be an closed-over constant
  outputs_flat <- lapply(output[[2L]], maybe_box_variable)

  desc@out_tree <- out_tree
  desc@outputs <- lapply(outputs_flat, \(x) x@gval)

  if (any(vapply(outputs_flat, \(x) !is_graph_box(x), logical(1L)))) {
    cli_abort("Function .f must return only objects of type `GraphBox`.")
  }

  graph <- descriptor_to_graph(desc)
  pass_dead_code(graph)
}

is_graph_node <- function(x) {
  is_graph_value(x) || is_graph_literal(x)
}

is_graph_value <- function(x) {
  inherits(x, "anvil::mut<GraphValue>")
}

maybe_restore_previous_desc <- function(graph = NULL) {
  if (!is.null(graph) && !identical(graph, globals[["CURRENT_DESCRIPTOR"]])) {
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
#' @return A [`Graph`] object.
#' @export
.current_descriptor <- function() {
  globals[["CURRENT_DESCRIPTOR"]] %??%
    cli_abort("No graph is currently being built")
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
#' @return A [`Graph`] object.
#' @export
local_descriptor <- function(..., envir = parent.frame()) {
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

graph_call <- function(prim, args, params = list()) {
  boxes_in <- lapply(args, maybe_box_variable)
  gvals_in <- lapply(boxes_in, \(x) x@gval)
  avals_in <- lapply(boxes_in, aval)

  vts_in <- lapply(avals_in, \(aval) st2vt(aval))
  outputs <- rlang::exec(prim[["graph"]], !!!c(vts_in, params))
  sts_out <- lapply(outputs, vt2st)

  gvals_out <- lapply(sts_out, GraphValue)

  call <- PrimitiveCall(prim, params, gvals_in, gvals_out)

  desc <- .current_descriptor()
  desc@calls <- c(desc@calls, call)
  boxes_out <- lapply(gvals_out, register_gval, desc = desc)

  return(boxes_out)
}


inline_graph_into_desc <- function(desc, graph) {
  for (const in graph@constants) {
    # The following can happen:
    # 1. a constant is already present in the parent descriptor -> do nothing
    # 2. the constant is not present in the parent descriptor -> register it
    get_box_or_register_cont(desc, const)
  }
  for (input in graph@inputs) {
    if (is.null(desc@gval_to_box[[input]])) {
      #
    }
    get_box_or_register_cont(desc, input)
  }

  desc@calls <- c(desc@calls, graph@calls)

  gvals_out_flat <- graph@outputs
  boxes_out_flat <- lapply(gvals_out_flat, GraphBox, desc)
  unflatten(graph@out_tree, boxes_out_flat)
}
