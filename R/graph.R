#' @title Graph Variable
#' @description
#' Variable in a [`Graph`].
#' @section Fields:
#' * `state` :: (`any`)\cr
#'   The state of the variable. Populated when the graph is executed.
#' @include mut.R
GraphVariable <- mut(new_class(
  "GraphVariable",
  properties = list(
    aval = ShapedTensor
  )
))

method(format, GraphVariable) <- function(x, ...) {
  sprintf("GraphVariable(%s)", format(x@aval))
}

GraphConstant <- mut(new_class(
  "GraphConstant",
  properties = list(
    aval = ConcreteTensor
  )
  # TODO: Why does fwd_graph not have inputs?
))

method(format, GraphConstant) <- function(x, ...) {
  sprintf("GraphConstant(%s)", format(x@aval))
}

GraphNode <- S7::new_union(GraphVariable, GraphConstant)

#' @title Primitive Call
#' @description
#' Call of a primitive in a [`Graph`]
#' Note that a primitive call also be a call into another graph (`p_graph`).
#' @section Fields:
#' * `primitive` :: ([`Primitive`])\cr
#'   The function.
#' * `inputs` :: (`list(GraphVariable)`)\cr
#'   The (tensor) inputs to the primitive.
#' * `params` :: (`list(<any>)`)\cr
#'   The (static) parameters of the function call.
#' * `outputs` :: (`list(GraphVariable)`)\cr
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
#' * `inputs` :: (`list(GraphVariable)`)\cr
#'   The inputs to the graph.
#' * `outputs` :: (`list(GraphVariable)`)\cr
#'   The outputs of the graph.
#'
#' @export
Graph <- mut(new_class(
  "Graph",
  properties = list(
    # Primitive: list(GraphVariable) --[params]--> list(GraphVariable)
    # All the GraphVariables that are inputs will already have a binding.
    # Those that are constants as well
    calls = list_of(PrimitiveCall),
    ## Used to (un-)flatten inputs and outputs
    in_tree = NULL | new_S3_class("Node"),
    out_tree = NULL | new_S3_class("Node"),
    inputs = list_of(GraphVariable),
    outputs = list_of(GraphNode),
    constants = list_of(GraphConstant)
  )
))

GraphDescriptor <- mut(new_class(
  "GraphDescriptor",
  properties = list(
    calls = list_of(PrimitiveCall),
    # hashtab, because we will deduplicate them during tracing
    constants = new_property(class_hashtab, default = quote(hashtab())),
    #constants = new_property(class_hashtab, default = quote(hashtab())),
    # AnvilTensor -> GraphConstant
    tensor_to_const = new_property(class_hashtab, default = quote(hashtab())),
    in_tree = NULL | new_S3_class("Node"),
    out_tree = NULL | new_S3_class("Node"),
    inputs = list_of(GraphVariable),
    outputs = list_of(GraphNode)
  )
))

method(shape, GraphVariable) <- function(x, ...) {
  shape(x@aval)
}

method(dtype, GraphVariable) <- function(x, ...) {
  dtype(x@aval)
}

method(shape, GraphConstant) <- function(x, ...) {
  shape(x@aval)
}

method(dtype, GraphConstant) <- function(x, ...) {
  dtype(x@aval)
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
    constants = hashvalues(descriptor@tensor_to_const)
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
    gvar = GraphNode,
    desc = GraphDescriptor
  )
)

method(shape, GraphBox) <- function(x, ...) {
  shape(x@gvar)
}

method(dtype, GraphBox) <- function(x, ...) {
  dtype(x@gvar)
}

method(print, GraphBox) <- function(x, ...) {
  cat(format(x), "\n")
}

method(format, GraphBox) <- function(x, ...) {
  sprintf("GraphBox(%s)", format(x@gvar))
}

aval <- function(x) {
  if (is_anvil_tensor(x)) {
    return(ConcreteTensor(x))
  }
  if (is_graph_box(x)) {
    return(x@gvar@aval)
  }
  cli_abort("internal error")
}


# TODO(aesthetics): can we unify maybe_box_variable and maybe_box_input?

maybe_box_variable <- function(x) {
  if (is_graph_box(x)) {
    if (x@desc == .current_descriptor()) {
      x
    } else {
      # closed over constants in nested graphify calls
      cli::cli_abort("Descriptor mismatch! x@desc: {format(x@desc)} current: {format(.current_descriptor())}")
    }
  } else if (is_anvil_tensor(x)) {
    desc <- .current_descriptor()

    const <- desc@tensor_to_const[[x]]

    if (!is.null(const)) {
      return(GraphBox(const, desc))
    }

    const <- GraphConstant(aval = ConcreteTensor(x))
    register_constant(desc, x, const)
    GraphBox(const, .current_descriptor())
  } else if (is_graph_node(x)) {
    GraphBox(x, .current_descriptor())
  } else {
    x
  }
}

register_constant <- function(desc, tensor, const) {
  desc@tensor_to_const[[tensor]] <- const
  desc@constants[[const]] <- const
}

initialize_descriptor_from_graph <- function(desc, graph) {
  desc@calls <- graph@calls
  for (const in graph@constants) {
    desc@constants[[const]] <- const
    desc@tensor_to_const[[const@aval@data]] <- const
  }
  desc@in_tree <- graph@in_tree
  desc@inputs <- graph@inputs
  desc@outputs <- graph@outputs
  desc@out_tree <- graph@out_tree

  graph
}

maybe_box_input <- function(x, desc) {
  if (is_anvil_tensor(x)) {
    # top-level graphify call
    gvar <- GraphVariable(aval = ShapedTensor(dtype(x), Shape(shape(x))))
    desc@inputs <- c(desc@inputs, gvar)
    GraphBox(gvar, desc)
  } else if (is_graph_box(x)) {
    # Nested graphify call
    # Because we will inline the child graph into the parent graph, we re-use
    # the same box, because this will make the inlining straightforward.
    desc@inputs <- c(desc@inputs, x@gvar)
    GraphBox(x@gvar, desc)
  } else {
    x
  }
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
  desc@outputs <- lapply(outputs_flat, \(x) x@gvar)

  if (any(vapply(outputs_flat, \(x) !is_graph_box(x), logical(1L)))) {
    cli_abort("Function .f must return only objects of type `GraphBox`.")
  }

  descriptor_to_graph(desc)
}

is_graph_node <- function(x) {
  is_graph_variable(x) || is_graph_constant(x)
}

is_graph_constant <- function(x) {
  inherits(x, "anvil::mut<GraphConstant>")
}

is_graph_variable <- function(x) {
  inherits(x, "anvil::mut<GraphVariable>")
}

# TODO: Nicer
#method(format, PrimitiveCall) <- function(x, ...) {
#  inputs <- paste(lapply(x@inputs, \(x) format(x@aval)), collapse = ", ")
#  outputs <- paste(lapply(x@outputs, \(x) format(x@aval)), collapse = ", ")
#  sprintf("%s(%s, %s params) -> %s", x@primitive@name, inputs, length(x@params), outputs)
#}

# This is like hlo_return, but for graphs
# It discards the current graph
#graph_return <- function(graph, outputs, out_tree) {
#  graph@outputs <- outputs
#  graph@out_tree <- out_tree
#  maybe_restore_previous_desc(graph)
#  graph
#}

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


eval_primitive_call <- function(call, mode, env) {
  in_vals <- lapply(call@inputs, \(x) env[[x]])
  out_vals <- do.call(call@primitive[[mode]], in_vals)

  for (i in seq_along(out_vals)) {
    env[[call@outputs[[i]]]] <- out_vals[[i]]
  }
}

graph_reduce <- function(graph, reducer, args) {
  env <- hashtab()

  args_flat <- flatten(args)

  if (length(args_flat) != length(graph@inputs)) {
    cli_abort("Expected {length(graph@inputs)} arguments, but got {length(args_flat)}")
  }

  # bind inputs
  for (i in seq_along(args_flat)) {
    env[[graph@inputs[[i]]]] <- args_flat[[i]]
  }

  for (call in graph@calls) {
    inputs <- lapply(call@inputs, \(x) env[[x]])
    output_vals <- reducer(call@primitive, inputs, call@params)
    for (i in seq_along(output_vals)) {
      env[[call@outputs[[i]]]] <- output_vals[[i]]
    }
  }

  lapply(graph@outputs, \(x) env[[x]])
}

graph_inline <- function(parent_desc, graph) {
  # TODO: constants
  parent_desc@calls <- c(parent_desc@calls, graph@calls)
  parent_desc
}

is_graph <- function(x) {
  inherits(x, "anvil::mut<Graph>")
}

#' @title Graph Transformation
#' @description
#' Abstract base class for a (chained) graph transformation.
#' To apply such a (chained) transformation, use [`apply_transform()`].
#' @export
GraphTransformation <- new_class(
  "GraphTransformation",
  parent = Transformation,
  properties = list(
    # TODO: Remove GraphTransformation here?
    input = class_function | Graph | new_class("GraphTransformation")
  )
)

is_graph_transformation <- function(x) {
  inherits(x, "anvil::GraphTransformation")
}

#' @title Transform a graph
#' @description
#' Apply a given transformation using the provided inputs.
#'
#' @section Adding a new Transformation:
#' In order to create a new transformatin, you need to:
#' 1. Create a custom subclass of `GraphTransformation`.
#' 1. Create a transformation function (like `gradient()`) that returns this subclass.
#' 1. Implement the `apply_transform()` method for this subclass.
#'    This is where the real complexity lies.
#'    Here, you can assume that the `@inputs` field is a [`Graph`], as recursion and conversion
#'    from a `function` to a [`Graph`] is handled by the generic.
#'
#' **Modifying Inputs or Outputs**:
#' When implementing a new transformation that modified the inputs or outputs of the graph,
#' it's important to ensure that `in_tree` and `out_tree` are updated accordingly.
#'
#' TODO:
#' @param x (`GraphTransformation`)\cr
#'   The transformation to apply.
#' @param args (`list(<any>)`)\cr
#'   The inputs to the transformation.
#' @return (`list(GraphTransformation | Graph, list(<any>))`)\cr
#'   The transformed graph(-transformation) and (possibly transformed) input values.
#' @export
apply_transform <- new_generic("transform", "gt", function(gt, args) {
  if (is_transformation(gt@input)) {
    Recall(do.call(apply_transform, gt@input), args)
  }
  # TODO(optional, cleaner): Maybe graphify should instead be inserted as its own transformation
  if (!is_graph(gt@input)) {
    out <- graphify(gt@input, args)
    gt@input <- out[[1L]]
    args <- out[[2L]]
  }
  S7::S7_dispatch()
})

GraphInterpreter <- new_class(
  "GraphInterpreter",
  parent = Interpreter,
  properties = list(
    graph = Graph
  )
)


is_graph_box <- function(x) {
  inherits(x, "anvil::GraphBox")
}

graph_call <- function(prim, args, params = list()) {
  boxes_in <- lapply(args, maybe_box_variable)
  gvars_in <- lapply(boxes_in, \(x) x@gvar)
  avals_in <- lapply(boxes_in, aval)

  vts_in <- lapply(avals_in, \(aval) st2vt(aval))
  outputs <- rlang::exec(prim[["graph"]], !!!c(vts_in, params))
  sts_out <- lapply(outputs, vt2st)

  gvars_out <- lapply(sts_out, GraphVariable)

  call <- PrimitiveCall(prim, params, gvars_in, gvars_out)

  graph <- .current_descriptor()
  graph@calls <- c(graph@calls, call)

  boxes_out <- lapply(gvars_out, \(gvar) {
    GraphBox(gvar, .current_descriptor())
  })

  return(boxes_out)
}


inline_graph_into_desc <- function(desc, graph) {
  desc@calls <- c(desc@calls, graph@calls)
  for (const in graph@constants) {
    desc@constants[[const]] <- const
    val <- const@aval@data
    if (!is_anvil_tensor(val)) {
      cli_abort("Constant {format(const)} is not an Anvil tensor")
    }
    desc@tensor_to_const[[val]] <- const
  }
  #desc@inputs <- c(desc@inputs, graph@inputs)
  #desc@outputs <- c(desc@outputs, graph@outputs)

  # they are graph variables
  gvars_out_flat <- graph@outputs
  boxes_out_flat <- lapply(gvars_out_flat, GraphBox, .current_descriptor())

  unflatten(graph@out_tree, boxes_out_flat)
}
