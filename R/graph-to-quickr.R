#' @include graph-to-r.R
NULL

#' Convert a Graph to a quickr-compiled function
#'
#' Convenience wrapper around [graph_to_r_function()] that inserts a
#' `declare(type(...))` header and compiles the resulting function with
#' `quickr::quick()`.
#'
#' @param graph ([`Graph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @export
graph_to_quickr_function <- function(graph) {
  assert_quickr_installed("{.fn graph_to_quickr_function}")
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls anvil::Graph}")
  }
  if (!inherits(graph@out_tree, "LeafNode") || length(graph@outputs) != 1L) {
    cli_abort("{.fn graph_to_quickr_function} currently supports only a single (non-list) output")
  }

  f <- graph_to_r_function(graph, include_declare = TRUE)
  quickr_eager_compile(f)
}
