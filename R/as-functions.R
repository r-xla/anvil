#' @include graph-to-quickr.R
NULL

#' Convert to an R function
#'
#' Convenience wrapper around [graph_to_r_function()] that always includes the
#' `declare(type(...))` header.
#'
#' @param graph ([`Graph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @export
as_r_function <- function(graph) {
  graph_to_r_function(graph, include_declare = TRUE)
}

#' Convert to a quickr-compiled function
#'
#' Convenience wrapper around [graph_to_quickr_function()].
#'
#' @param graph ([`Graph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @export
as_quickr_function <- function(graph) {
  graph_to_quickr_function(graph)
}
