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
#' @param constants (`character(1)`)\cr
#'   How to handle `graph@constants` (closed-over tensors):
#'   - `"inline"`: embed them as R literals in the function body (default).
#'   - `"args"`: add them as additional function arguments (named `c1`, `c2`, ...).
#' @return (`function`)
#' @export
graph_to_quickr_function <- function(graph, constants = c("inline", "args")) {
  constants <- match.arg(constants)
  if (!requireNamespace("quickr", quietly = TRUE)) {
    cli_abort("{.pkg quickr} must be installed to use {.fn graph_to_quickr_function}")
  }
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls anvil::Graph}")
  }
  if (!inherits(graph@out_tree, "LeafNode") || length(graph@outputs) != 1L) {
    cli_abort("{.fn graph_to_quickr_function} currently supports only a single (non-list) output")
  }

  f <- graph_to_r_function(graph, constants = constants, include_declare = TRUE)

  # quickr::quick() behaves differently when called from a package namespace
  # (it creates a closure expecting precompiled artifacts). For anvil's use-case
  # we want eager compilation at runtime, so we evaluate quickr::quick() from a
  # non-namespace environment.
  tmp <- new.env(parent = globalenv())
  tmp$fun <- f
  eval(quote(quickr::quick(fun)), envir = tmp)
}
