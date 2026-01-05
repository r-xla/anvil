traverse_gnodes <- function(graph, fn, graph_outputs = TRUE) {
  for (call in graph@calls) {
    for (input in call@inputs) {
      fn(input)
    }
    if (is_higher_order_primitive(call@primitive)) {
      lapply(subgraphs(call), traverse_gnodes, fn = fn, graph_outputs = graph_outputs)
    }
  }
  if (graph_outputs) {
    for (output in graph@outputs) {
      fn(output)
    }
  }
}

remove_unused_constants <- function(graph) {
  is_used <- hashtab()
  # here we assume that higher-order primitives capture their constants via
  # lexical scoping and don't have constants of their own
  # this means, the main graph contains all the constants that are used
  traverse_gnodes(graph, function(gval) {
    if (is_graph_value(gval) && is_concrete_tensor(gval@aval)) {
      is_used[[gval]] <- TRUE
    }
  })
  graph@constants <- graph@constants[vapply(graph@constants, function(const) isTRUE(is_used[[const]]), logical(1L))]
  graph
}

inline_scalarish_constants <- function(graph) {
  is_scalarish <- function(gval) {
    is_graph_value(gval) && is_concrete_tensor(gval@aval) && (prod(gval@aval@shape@dims) == 1L)
  }

  scalarish_to_lit <- function(gval) {
    GraphLiteral(LiteralTensor(as_array(gval@aval@data), shape = shape(gval@aval), dtype = dtype(gval@aval)))
  }

  map <- hashtab()
  for (const in graph@constants) {
    if (is_scalarish(const)) {
      browser()
      map[[const]] <- scalarish_to_lit(const)
    }
  }
  for (i in seq_along(graph@calls)) {
    pcall <- graph@calls[[i]]
    for (j in seq_along(pcall@inputs)) {
      if (is_scalarish(pcall@inputs[[j]])) {
        graph@calls[[i]]@inputs[[j]] <- map[[pcall@inputs[[j]]]]
      }
    }
    if (is_higher_order_primitive(pcall@primitive)) {
      for (subgraph in subgraphs(pcall)) {
        subgraph <- inline_scalarish_constants(subgraph)
      }
    }
  }
}
