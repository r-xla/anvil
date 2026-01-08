traverse_gnodes <- function(graph, fn, graph_outputs = TRUE) {
  for (call in graph$calls) {
    for (input in call$inputs) {
      fn(input)
    }
    if (is_higher_order_primitive(call$primitive)) {
      lapply(subgraphs(call), traverse_gnodes, fn = fn, graph_outputs = graph_outputs)
    }
  }
  if (graph_outputs) {
    for (output in graph$outputs) {
      fn(output)
    }
  }
}

remove_unused_constants <- function(graph) {
  new_graph <- Graph(
    calls = graph$calls,
    in_tree = graph$in_tree,
    out_tree = graph$out_tree,
    inputs = graph$inputs,
    outputs = graph$outputs,
    constants = graph$constants
  )

  is_used <- hashtab()
  # here we assume that higher-order primitives capture their constants via
  # lexical scoping and don't have constants of their own
  # this means, the main graph contains all the constants that are used
  traverse_gnodes(new_graph, function(gval) {
    if (is_graph_value(gval) && is_concrete_tensor(gval$aval)) {
      is_used[[gval]] <- TRUE
    }
  })
  new_graph$constants <- new_graph$constants[vapply(
    new_graph$constants,
    function(const) isTRUE(is_used[[const]]),
    logical(1L)
  )]
  new_graph
}

inline_scalarish_constants <- function(graph, map = NULL) {
  is_scalarish <- function(gval) {
    is_graph_value(gval) && is_concrete_tensor(gval$aval) && (prod(gval$aval$shape$dims) == 1L)
  }

  scalarish_to_lit <- function(gval) {
    GraphLiteral(LiteralTensor(
      gval$aval$data,
      shape = shape(gval$aval),
      dtype = dtype(gval$aval),
      ambiguous = gval$aval$ambiguous
    ))
  }

  # Create a copy of the graph
  new_graph <- Graph(
    calls = graph$calls,
    in_tree = graph$in_tree,
    out_tree = graph$out_tree,
    inputs = graph$inputs,
    outputs = graph$outputs,
    constants = graph$constants
  )

  is_top_level <- is.null(map)
  map <- map %||% hashtab()
  for (const in new_graph$constants) {
    if (is_scalarish(const)) {
      map[[const]] <- scalarish_to_lit(const)
    }
  }
  for (i in seq_along(new_graph$inputs)) {
    replacement <- map[[new_graph$inputs[[i]]]]
    if (!is.null(replacement)) {
      new_graph$inputs[[i]] <- replacement
    }
  }

  for (i in seq_along(new_graph$calls)) {
    pcall <- new_graph$calls[[i]]
    for (j in seq_along(pcall$inputs)) {
      replacement <- map[[pcall$inputs[[j]]]]
      if (!is.null(replacement)) {
        new_graph$calls[[i]]$inputs[[j]] <- replacement
      }
    }
    if (is_higher_order_primitive(pcall$primitive)) {
      subgraph_names <- pcall$primitive$subgraphs
      for (name in subgraph_names) {
        if (name %in% names(pcall$params)) {
          new_subgraph <- inline_scalarish_constants(pcall$params[[name]], map)
          new_graph$calls[[i]]$params[[name]] <- new_subgraph
        }
      }
    }
  }
  for (i in seq_along(new_graph$outputs)) {
    replacement <- map[[new_graph$outputs[[i]]]]
    if (!is.null(replacement)) {
      new_graph$outputs[[i]] <- replacement
    }
  }
  # TODO: We could ensure that each constant is only added once to the graph (currently, two
  # nv_scalar(1) will create to fill calls)
  if (is_top_level) {
    consts <- hashvalues(map)
    new_graph$calls <- c(
      new_graph$calls,
      lapply(consts, function(const) {
        PrimitiveCall(
          primitive = p_fill,
          inputs = list(),
          params = list(value = const$aval$data, dtype = dtype(const$aval), shape = shape(const$aval)),
          outputs = list(const)
        )
      })
    )
  }
  new_graph$constants <- new_graph$constants[vapply(
    new_graph$constants,
    function(const) is.null(map[[const]]),
    logical(1L)
  )]
  new_graph
}
