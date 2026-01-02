#' @include graph.R
#' @include primitive.R

#' @title Inline Literals Pass
#' @description
#' Converts all scalar constants (shape `c()`) in a graph to `GraphLiteral` objects.
#' This pass also recursively processes subgraphs in higher-order primitives.
#' @param graph (`Graph`)\cr
#'   The graph to process.
#' @return (`Graph`)\cr
#'   A new graph with scalar constants converted to literals.
#' @export
inline_literals <- function(graph) {
  if (!is_graph(graph)) {
    cli_abort("graph must be a Graph object")
  }

  # Create a mapping from old GraphValue constants to new GraphLiteral objects
  # Use hashtab to store object-to-object mappings
  replacement_map <- hashtab()

  # First pass: identify scalar constants and create replacements
  for (const in graph@constants) {
    if (!is_graph_value(const) || !is_concrete_tensor(const@aval)) {
      cli_abort("Internal error: Not all constants are concrete tensors")
    }
    const_shape <- shape(const@aval)
    if (!length(const_shape)) {
      scalar_value <- as_array(const@aval@data)
      # Create the new GraphLiteral
      literal_tensor <- LiteralTensor(
        data = scalar_value,
        shape = integer(),
        dtype = const@aval@dtype,
        #  ConcreteTensors should not be ambiguous anyway
        ambiguous = const@aval@ambiguous
      )
      new_literal <- GraphLiteral(aval = literal_tensor)

      # Store the replacement using the object itself as the key
      replacement_map[[const]] <- new_literal
    }
  }

  # Helper function to replace GraphValue with GraphLiteral if in map
  replace_node <- function(node) {
    if (is_graph_value(node)) {
      replacement <- replacement_map[[node]]
      if (!is.null(replacement)) {
        return(replacement)
      }
    }
    return(node)
  }

  # Second pass: replace references in calls
  new_calls <- lapply(graph@calls, function(call) {
    new_inputs <- lapply(call@inputs, replace_node)
    new_outputs <- lapply(call@outputs, replace_node)

    # Handle higher-order primitives: recursively process subgraphs
    new_params <- call@params
    if (is_higher_order_primitive(call@primitive)) {
      subgraphs_list <- subgraphs(call)
      # Process each subgraph and update the corresponding parameter
      for (subgraph in subgraphs_list) {
        processed_subgraph <- inline_literals(subgraph)
        # Find the parameter name that contains this subgraph
        for (param_name in names(call@params)) {
          if (identical(call@params[[param_name]], subgraph)) {
            new_params[[param_name]] <- processed_subgraph
            break
          }
        }
      }
    }

    PrimitiveCall(
      primitive = call@primitive,
      inputs = new_inputs,
      params = new_params,
      outputs = new_outputs
    )
  })

  # Replace references in outputs
  new_outputs <- lapply(graph@outputs, replace_node)

  # Remove converted constants from the constants list
  new_constants <- list()
  for (const in graph@constants) {
    replacement <- replacement_map[[const]]
    if (is.null(replacement)) {
      new_constants <- c(new_constants, list(const))
    }
  }

  new_graph <- Graph(
    calls = new_calls,
    inputs = graph@inputs,
    outputs = new_outputs,
    constants = new_constants,
    in_tree = graph@in_tree,
    out_tree = graph@out_tree
  )
  return(new_graph)
}
