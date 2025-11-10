# TODO: What about constants?
# We don't even need to keep a reference to the Graph, like in FuncVariable, because
# we can just access the global data of the main interpreter, where we need to store the graph anyway
# For functions that don't have any arguments (where we hence can't retrieve the graph from any of the input GraphVariables)
GraphVariable <- mut(new_class(
  "GraphVariable",
  properties = list(
    .env = class_hashtab,
    # We use this when executing a graph with actual values
    state = new_property(class_any),
    aval = new_property(ShapedTensor)
  ),
))

PrimitiveCall <- new_class(
  "PrimitiveCall",
  properties = list(
    primitive = Primitive,
    params = list_of(class_any),
    inputs = list_of(GraphVariable),
    outputs = list_of(GraphVariable)
  )
)

#' @include mut.R
GraphBuilder <- mut(new_class("GraphBuilder",
  properties = list(
    calls = list_of(PrimitiveCall),
    in_tree = NULL | class_node,
    out_tree = NULL | class_node,
    abstract_inputs = list_of(ShapedTensor),
    gvars = list_of(GraphVariable),
    consts_to_gvars = class_hashtab,
    consts = list_of(ShapedTensor),
    output_vars = list_of(GraphVariable)
  )
))



#' @title Graph of Primitive Calls
#' @description
#' A graph consists of:
#' * `calls`: a `list()` of [`PrimitiveCall`] objects.
#' * `in_tree`: a `Node` for unflattening flattened inputs.
#' * `out_tree`: a `Node` for unflattening flat outputs.
#' * `abstract_inputs` : `list()` of `ShapedTensor` objects.
#' * `constants`: `list()` of `ConcreteTensor` objects.
#'
#' @noRd
Graph <- new_class(
  "Graph",
  properties = list(
    # Primitive: list(GraphVariable) --[params]--> list(GraphVariable)
    # All the GraphVariables that are inputs will already have a binding.
    # Those that are constants as well
    calls = list_of(PrimitiveCall),
    # Used to (un-)flatten inputs and outputs
    in_tree = NULL | new_S3_class("Node"),
    out_tree = NULL | new_S3_class("Node"),
    inputs = list_of(GraphVariable),
    outputs = list_of(GraphVariable)
  )
)

is_graph <- function(x) {
  inherits(x, "anvil::Graph")
}

#' @title Graph Transformation
#' @description
#' Base class for a graph transformation.
#' To conduct a such a graph transformation, input values need to be provided.
#' A recipe for a graph transformation.
GraphTransformation <- new_class("GraphTransformation",
  properties = list(
    func = class_function | new_S3_class("anvil::GraphTransformation")
  )
)

#' @title Transform a graph
#' @description
#' Transform a graph given some inputs.
#' @section Contract:
#' The
#' @param x (`GraphTransformation`)\cr
#'   The transformation to apply.
#' @param ... (`any`)\cr
#'   The inputs to the transformation.
#' @param .args (`list`)\cr
#'   Additional arguments to the transformation.
#' @return (`list(Graph, list(<any>))`\cr
#'   The transformed graph and (possibly transformed) input values.
#' @export
transform <- new_generic("transform", "x", function(x, ..., .args) {
  x <- graphify(x, ...)
  S7::S7_dispatch()
})

GraphInterpreter <- new_class(
  "GraphInterpreter",
  parent = Interpreter,
  properties = list(
    graph = Graph
  )
)


GraphBox <- new_class(
  "GraphBox",
  parent = Box,
  properties = list(
    graph_variable = GraphVariable,
    primal = class_any
  )
)


append_primitive_call <- function(graph, call) {
  graph@calls <- c(graph@calls, call)
  graph
}

method(process_primitive, GraphInterpreter) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  avals_in <- lapply(boxes, aval)
  outputs <- rlang::exec(prim[["graph"]], !!!c(avals_in, params))

  gvars_out <- lapply(outputs, GraphVariable)
  gvars_in <- lapply(boxes, \(b) b@graph_variable)


  # in-place modification
  append_primitive_call(interpreter@main@global_data,
    PrimitiveCall(prim, params, gvars_in, gvars_out)
  )

  lapply(gvars_out, GraphBox, interpreter = interpreter)
}


graphify <- function(.f, ...) {
  # TODO: flattening
  args <- list(...)
  in_tree <- build_tree(list(...))
  f_flat <- flatten_fun(f, in_node = in_tree)
  graph <- Graph(in_tree = in_tree)

  graph <- Graph(in_tree = in_tree)
  main <- local_main(GraphInterpreter, global_data = graph)
  interpreter <- GraphInterpreter(main = main)
  args <- lapply(args, full_raise, interpreter = interpreter)
  out <- do.call(f_flat, args)
  graph@out_tree <- out[[1L]]
  graph@output_vars <- lapply(out[[2L]], \(x) x@graph_variable)
  return(graph)
}
# Boxing only happens during graphify.
# Afterwards, all variables that need to be boxed are the input(-constants)

method(box, list(GraphInterpreter, ShapedTensor)) <- function(interpreter, x) {
  GraphBox(interpreter, GraphVariable(aval = x))
}

method(box, list(GraphInterpreter, class_any)) <- function(interpreter, x) {
  box(interpreter, aval(x))
}

method(aval, GraphBox) <- function(x) {
  x@graph_variable@aval
}


method(format, GraphBox) <- function(x) {
  sprintf("GraphBox(%s)", format(x@graph_variable@aval))
}

method(print, GraphBox) <- function(x) {
  cat(format(x), "\n")
}
