GraphBox <- new_class(
  "GraphBox",
  parent = Box,
  properties = list(
    graph = Graph
  )
)

GraphInterpreter <- new_class(
  "GraphInterpreter",
  parent = Interpreter,
  properties = list(
    graph = Graph
  )
)

PrimitiveCall <- new_class(
  "PrimitiveCall",
  properties = list(
    primitive = Primitive,
    args = list_of(class_any)
  )
)

#' @title Graph
#' @description
#' This is a computational graph.
#' The nodes are anvil primitives and the edges are the connections between them.
#' @noRd
Graph <- new_class(
  "Graph",
  properties = list(
    nodes = list_of(Primitive),
    edges = list_of(new_class("GraphEdge"))
  )
)

GraphNode <- new_class(
  "Node",
  properties = list(
    .environment = class_environment
  ),
  constructor = function() {
    S7::new_object(S7::S7_object(), .environment = new.env(size = 0L, parent = emptyenv()))
  }
)

GraphEdge <- new_class(
  "GraphEdge",
  properties = list(
    id = class_integer
  )
)


method(process_primitive, GraphInterpreter) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  nodes <- list()
  edges <- list()
  for (box in boxes) {
    nodes <- c(nodes, box@node)
    edges <- c(edges, box@edge)
  }
  prim[["graph"]](nodes, edges, params)
  list(GraphBox(interpreter, Graph(nodes = nodes, edges = edges)))
}
