# The graph rules are very simple, we do no type inference, but just build the graph and
# make closed over constants explicit, which is e.g. important for the pullback rule.

p_add[["graph"]] <- function(in_nodes, graph) {
  node <- GraphNode()
  add_node(graph, node)
  add_edge(graph, in_nodes[[1L]], node)
  add_edge(graph, in_nodes[[2L]], node)
}



f <- function(x, y) {
  x + y
}

graph <- interprete(f, GraphInterpreter, list(nv_scalar(1), nv_scalar(2)))

backward_graph <- interprete(graph, PullbackInterpreter, list(nv_scalar(1)))
