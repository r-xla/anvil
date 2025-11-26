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

remove_unused_constants <- function(graph) {
  live <- hashtab()
  for (node in graph@outputs) {
    live[[node]] <- TRUE
  }

  for (call in rev(graph@calls)) {
    any_output_live <- any(vapply(
      call@outputs,
      function(out) {
        isTRUE(live[[out]])
      },
      logical(1L)
    ))
  }
}
