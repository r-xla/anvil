#' Remove calls from a graph that do not contribute to the outputs
#'
#' @param graph A Graph object
#' @return A new Graph with dead code removed
#' @noRd
remove_dead_code <- function(graph) {
  if (length(graph@calls) == 0L) {
    return(graph)
  }

  # Start with outputs as live nodes
  live <- hashtab()
  for (node in graph@outputs) {
    live[[node]] <- TRUE
  }

  # Walk backwards through calls
  # If any output of a call is live, all its inputs become live
  for (call in rev(graph@calls)) {
    any_output_live <- any(vapply(
      call@outputs,
      function(out) {
        isTRUE(live[[out]])
      },
      logical(1L)
    ))

    if (any_output_live) {
      for (input in call@inputs) {
        live[[input]] <- TRUE
      }
    }
  }

  #
  # We don't remove constants, because they might be needed
  # by closures (nv_if etc.)
  #if (unused_constants) {
  #  # Filter constants: keep only those that are live
  #  live_constants <- Filter(function(const) {
  #    isTRUE(live[[const]])
  #  }, graph@constants)
  #  graph@constants <- live_constants
  #}

  # Filter calls: keep only those with at least one live output
  live_calls <- Filter(
    function(call) {
      any(vapply(
        call@outputs,
        function(out) {
          isTRUE(live[[out]])
        },
        logical(1L)
      ))
    },
    graph@calls
  )

  graph@calls <- live_calls
  graph
}

fuse_broadcast_in_dim_mul_ops <- function(ops) {
  is_same_sym <- function(expr, sym) {
    is.name(expr) && identical(expr, sym)
  }

  count_sym_uses <- function(ops, sym) {
    n <- 0L
    for (op in ops) {
      for (inp in op$inputs) {
        if (is_same_sym(inp, sym)) n <- n + 1L
      }
    }
    n
  }

  out <- list()
  i <- 1L
  n <- length(ops)
  while (i <= n) {
    op <- ops[[i]]
    if (op$prim_name == "broadcast_in_dim" && i < n && length(op$out_syms) == 1L) {
      b_sym <- op$out_syms[[1L]]
      if (count_sym_uses(ops, b_sym) == 1L) {
        next_op <- ops[[i + 1L]]
        if (next_op$prim_name == "mul" && length(next_op$out_syms) == 1L) {
          b_pos <- which(vapply(next_op$inputs, is_same_sym, logical(1L), sym = b_sym))
          if (length(b_pos) == 1L) {
            other_pos <- if (b_pos == 1L) 2L else 1L

            y_node <- op$input_nodes[[1L]]
            if (is_graph_value(y_node) && length(shape(y_node@aval)) == 1L) {
              shape_out <- op$params$shape_out
              bd <- as.integer(op$params$broadcast_dimensions)
              if (length(bd) == 1L) {
                axis <- bd[[1L]]
                ok_axis <- axis >= 1L && axis <= length(shape_out)
                ok_len <- as.integer(shape_out[[axis]]) == as.integer(shape(y_node@aval)[[1L]])

                x_node <- next_op$input_nodes[[other_pos]]
                if (ok_axis && ok_len && is_graph_value(x_node) && identical(shape(x_node@aval), shape_out)) {
                  fused <- list(
                    prim_name = "mul_broadcast_axis",
                    inputs = list(next_op$inputs[[other_pos]], op$inputs[[1L]]),
                    params = list(axis = axis, shape_out = shape_out),
                    out_syms = next_op$out_syms,
                    input_nodes = next_op$input_nodes,
                    out_avals = next_op$out_avals
                  )
                  out <- c(out, list(fused))
                  i <- i + 2L
                  next
                }
              }
            }
          }
        }
      }
    }
    out <- c(out, list(op))
    i <- i + 1L
  }
  out
}
