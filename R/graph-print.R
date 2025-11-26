#' @include graph.R

format_node_id <- function(node, node_ids) {
  id <- node_ids[[node]]
  if (is.null(id)) {
    return("???")
  }
  sprintf("%%%s", id)
}

# Format aval as type signature
format_aval_short <- function(aval) {
  sprintf("%s[%s]", repr(dtype(aval)), paste(shape(aval), collapse = ", "))
}

# Build a mapping from nodes to unique IDs
build_node_ids <- function(inputs, constants, calls) {
  node_ids <- hashtab()
  counter <- 0L

  # Inputs get sequential IDs: a, b, c, ...
  for (node in inputs) {
    # FIXME: !!!
    node_ids[[node]] <- letters[counter + 1L]
    counter <- counter + 1L
  }

  # Constants get c0, c1, c2, ...
  const_counter <- 0L
  for (node in constants) {
    node_ids[[node]] <- sprintf("c%d", const_counter)
    const_counter <- const_counter + 1L
  }

  # Outputs of calls get numeric IDs: 0, 1, 2, ...
  num_counter <- 0L
  for (call in calls) {
    for (out in call@outputs) {
      node_ids[[out]] <- as.character(num_counter)
      num_counter <- num_counter + 1L
    }
  }

  node_ids
}

# Format a single PrimitiveCall given a node_id mapping
format_call <- function(call, node_ids, indent = "  ") {
  # Format inputs
  input_ids <- vapply(call@inputs, format_node_id, character(1), node_ids = node_ids)
  inputs_str <- paste(input_ids, collapse = ", ")

  # Format outputs
  output_ids <- vapply(call@outputs, format_node_id, character(1), node_ids = node_ids)
  output_types <- vapply(call@outputs, \(x) format_aval_short(x@aval), character(1))

  if (length(call@outputs) == 1L) {
    outputs_str <- sprintf("%s: %s", output_ids, output_types)
  } else {
    outputs_str <- sprintf("(%s): (%s)", paste(output_ids, collapse = ", "), paste(output_types, collapse = ", "))
  }

  # Format params if present
  params_str <- sprintf(" [%d params]", length(call@params))
  #if (length(call@params) > 0L) {
  #  param_parts <- vapply(seq_along(call@params), function(i) {
  #    val <- call@params[[i]]
  #    nm <- names(call@params)[i]
  #    # Format vectors/lists nicely
  #    val_str <- if (is.atomic(val) && length(val) > 1L) {
  #      sprintf("c(%s)", paste(val, collapse = ", "))
  #    } else if (is.list(val)) {
  #      deparse1(val)
  #    } else {
  #      as.character(val)
  #    }
  #    if (is.null(nm) || nm == "") {
  #      val_str
  #    } else {
  #      sprintf("%s=%s", nm, val_str)
  #    }
  #  }, character(1))
  #  params_str <- sprintf("{%s}", paste(param_parts, collapse = ", "))
  #}

  sprintf("%s%s = %s%s(%s)", indent, outputs_str, call@primitive@name, params_str, inputs_str)
}

# Main formatting function for graph-like structures
format_graph_body <- function(inputs, constants, calls, outputs, title = "Graph") {
  lines <- character()

  # Build node ID mapping
  node_ids <- build_node_ids(inputs, constants, calls)

  # Header
  lines <- c(lines, sprintf("<%s>", title))

  # Inputs section
  if (length(inputs) > 0L) {
    input_strs <- vapply(
      inputs,
      function(node) {
        sprintf("    %s: %s", format_node_id(node, node_ids), format_aval_short(node@aval))
      },
      character(1)
    )
    lines <- c(lines, "  Inputs:", input_strs)
  } else {
    lines <- c(lines, "  Inputs: (none)")
  }

  # Constants section
  if (length(constants) > 0L) {
    const_strs <- vapply(
      constants,
      function(node) {
        sprintf("    %s: %s", format_node_id(node, node_ids), format_aval_short(node@aval))
      },
      character(1)
    )
    lines <- c(lines, "  Constants:", const_strs)
  }

  # Calls section
  if (length(calls) > 0L) {
    lines <- c(lines, "  Body:")
    for (call in calls) {
      lines <- c(lines, format_call(call, node_ids, indent = "    "))
    }
  } else {
    lines <- c(lines, "  Body: (empty)")
  }

  # Outputs section
  if (length(outputs) > 0L) {
    output_strs <- vapply(
      outputs,
      function(node) {
        sprintf("    %s: %s", format_node_id(node, node_ids), format_aval_short(node@aval))
      },
      character(1)
    )
    lines <- c(lines, "  Outputs:", output_strs)
  } else {
    lines <- c(lines, "  Outputs: (none)")
  }

  paste(lines, collapse = "\n")
}

method(format, PrimitiveCall) <- function(x, ...) {
  inputs <- paste(vapply(x@inputs, \(inp) format_aval_short(inp@aval), character(1)), collapse = ", ")
  outputs <- paste(vapply(x@outputs, \(out) format_aval_short(out@aval), character(1)), collapse = ", ")
  params_str <- if (length(x@params) > 0L) sprintf(" {%d params}", length(x@params)) else ""
  sprintf("%s(%s)%s -> %s", x@primitive@name, inputs, params_str, outputs)
}

method(format, Graph) <- function(x, ...) {
  format_graph_body(
    inputs = x@inputs,
    constants = x@constants,
    calls = x@calls,
    outputs = x@outputs,
    title = "Graph"
  )
}

method(print, Graph) <- function(x, ...) {
  cat(format(x), "\n")
}

method(format, GraphDescriptor) <- function(x, ...) {
  # Convert hashtab constants to list
  constants <- x@constants
  format_graph_body(
    inputs = x@inputs,
    constants = constants,
    calls = x@calls,
    outputs = x@outputs,
    title = "GraphDescriptor"
  )
}

method(print, GraphDescriptor) <- function(x, ...) {
  cat(format(x), "\n")
}
