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

  inner_fun <- graph_to_r_function(graph, include_declare = TRUE)
  inner_quick <- quickr_eager_compile(inner_fun)

  # If the graph's input tree contains nested list structure, graph_to_r_function()
  # flattens that into leaf arguments (because quickr can't take lists as args).
  # Wrap the compiled function in a plain R closure that accepts the original
  # top-level inputs and flattens them to the leaf args expected by inner_quick().
  in_tree <- graph@in_tree
  if (!inherits(in_tree, "ListNode")) {
    return(inner_quick)
  }

  top_nodes <- in_tree$nodes
  has_nested <- any(vapply(top_nodes, inherits, logical(1L), "ListNode"))
  if (!has_nested) {
    return(inner_quick)
  }

  top_names <- in_tree$names
  if (is.null(top_names) || any(!nzchar(top_names))) {
    top_names <- paste0("arg", seq_along(top_nodes))
  }
  top_names <- make.unique(make.names(top_names))

  make_formals <- function(nms) {
    as.pairlist(stats::setNames(rep(list(quote(expr = )), length(nms)), nms))
  }

  wrapper <- function() {
    stop("internal placeholder")
  }
  formals(wrapper) <- make_formals(top_names)

  inner_env <- environment()
  args_top_call <- as.call(c(list(as.name("list")), lapply(top_names, as.name)))
  body(wrapper) <- bquote({
    args_top <- .(args_top_call)
    args_flat <- flatten(args_top)
    do.call(.(inner_env$inner_quick), args_flat)
  })

  wrapper
}
