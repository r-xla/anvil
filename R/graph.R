# TODO: What about constants?
# We don't even need to keep a reference to the Graph, like in FuncVariable, because
# we can just access the global data of the main interpreter, where we need to store the graph anyway
# For functions that don't have any arguments (where we hence can't retrieve the graph from any of the input GraphVariables)
GraphVariable <- new_class(
  "GraphVariable",
  properties = list(
    .environment = class_environment,
    # We use this when executing a graph with actual values
    binding = NULL | class_any,
    value_type = ShapedTensor
  ),
  constructor = function(value_type) {
    S7::new_object(S7::S7_object(), .environment = zero_env(), value_type = value_type)
  }
)

PrimitiveCall <- new_class(
  "PrimitiveCall",
  properties = list(
    primitive = Primitive,
    params = list_of(class_any),
    inputs = list_of(GraphVariable),
    outputs = list_of(GraphVariable)
  )
)

#' @title Graph of Primitive Calls
#' @description
#' This graph is the basis on which things like jit and pullback work.
#' @noRd
Graph <- new_class(
  "Graph",
  properties = list(
    calls = new_property(
      getter = function(self) {
        self@.data[["calls"]]
      },
      setter = function(self, value) {
        self@.data[["calls"]] <- value
        invisible(self)
      }
    ),
    .data = new_S3_class("hashtab")
  ),
  constructor = function() {
    .data <- hashtab()
    .data[["calls"]] <- list()
    new_object(S7_object(), .data = .data)
  }
)

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

graph_interpret <- function(.interpreter, ...) {
  # 1. Bind inputs to input GraphVariables
  # 2. Iterate the expressions and bind the intermediate results to the bindings
  # 3. Compute the outputs
  # 4. Clean the bindings from the variables. We don't have to worry about memory issues
  #    because we only ever to abstract evaluation, so keeping previous compute results in
  #    memory is fine.


}

graph_inputs <- function(graph) {
  # TODO: cached impl
}


graph_outputs <- function(graph) {
  # TODO: cached impl
}


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
  main <- local_main(GraphInterpreter, global_data = Graph())
  interpreter <- GraphInterpreter(main = main)
  args <- lapply(args, full_raise, interpreter = interpreter)
  print(args)
  out <- do.call(.f, args)

  main@global_data
}

method(box, list(GraphInterpreter, ShapedTensor)) <- function(interpreter, x) {
  GraphBox(interpreter, GraphVariable(value_type = x))
}

method(box, list(GraphInterpreter, class_any)) <- function(interpreter, x) {
  box(interpreter, aval(x))
}

method(aval, GraphBox) <- function(x) {
  x@graph_variable@value_type
}


method(format, GraphBox) <- function(x) {
  sprintf("GraphBox(%s)", format(x@graph_variable@value_type))
}

method(print, GraphBox) <- function(x) {
  cat(format(x), "\n")
}
