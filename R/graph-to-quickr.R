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

  needs_pack <- !(inherits(graph@out_tree, "LeafNode") && length(graph@outputs) == 1L)

  inner_fun <- graph_to_r_function(graph, include_declare = TRUE, pack_output = needs_pack)
  inner_quick <- quickr_eager_compile(inner_fun)

  # If the graph's input tree contains nested list structure, graph_to_r_function()
  # flattens that into leaf arguments (because quickr can't take lists as args).
  # Wrap the compiled function in a plain R closure that accepts the original
  # top-level inputs and flattens them to the leaf args expected by inner_quick().
  in_tree <- graph@in_tree
  has_nested <- FALSE
  top_names <- NULL
  if (inherits(in_tree, "ListNode")) {
    top_nodes <- in_tree$nodes
    has_nested <- any(vapply(top_nodes, inherits, logical(1L), "ListNode"))
    if (has_nested) {
      top_names <- in_tree$names
      if (is.null(top_names) || any(!nzchar(top_names))) {
        top_names <- paste0("arg", seq_along(top_nodes))
      }
      top_names <- make.unique(make.names(top_names))
    }
  }

  if (!isTRUE(needs_pack) && !isTRUE(has_nested)) {
    return(inner_quick)
  }

  arg_names <- if (isTRUE(has_nested)) top_names else names(formals(inner_fun))

  make_formals <- function(nms) {
    as.pairlist(stats::setNames(rep(list(quote(expr = )), length(nms)), nms))
  }

  out_infos <- lapply(graph@outputs, function(node) {
    if (is_graph_value(node)) {
      list(dtype = as.character(dtype(node@aval)), shape = shape(node@aval))
    } else {
      list(dtype = as.character(node@dtype), shape = integer())
    }
  })
  out_lens <- vapply(out_infos, function(info) {
    if (!length(info$shape)) 1L else Reduce(`*`, as.integer(info$shape), init = 1L)
  }, integer(1L))

  wrapper <- function() {
    stop("internal placeholder")
  }
  formals(wrapper) <- make_formals(arg_names)

  wrapper_env <- new.env(parent = environment())
  wrapper_env$inner_quick <- inner_quick
  wrapper_env$needs_pack <- needs_pack
  wrapper_env$out_tree <- graph@out_tree
  wrapper_env$out_infos <- out_infos
  wrapper_env$out_lens <- out_lens
  wrapper_env$arg_names <- arg_names

  body(wrapper) <- quote({
    args_top <- as.list(match.call())[-1L]
    args_top <- lapply(args_top, eval, envir = parent.frame())
    args_top <- args_top[arg_names]
    args_flat <- flatten(args_top)
    packed <- do.call(inner_quick, args_flat)

    if (!isTRUE(needs_pack)) {
      return(packed)
    }

    decode_leaf <- function(seg, shape, dtype) {
      if (dtype %in% c("pred", "i1")) {
        base <- seg != 0
      } else if (grepl("^(u?i)(8|16|32|64)$", dtype)) {
        base <- as.integer(seg)
      } else {
        base <- as.double(seg)
      }

      if (!length(shape)) {
        return(base[[1L]])
      }
      if (length(shape) == 1L) {
        return(base)
      }
      if (length(shape) == 2L) {
        return(matrix(base, nrow = shape[[1L]], ncol = shape[[2L]]))
      }
      array(base, dim = shape)
    }

    leaves <- vector("list", length(out_infos))
    pos <- 0L
    for (i in seq_along(out_infos)) {
      len <- out_lens[[i]]
      seg <- packed[(pos + 1L):(pos + len)]
      pos <- pos + len
      leaves[[i]] <- decode_leaf(seg, out_infos[[i]]$shape, out_infos[[i]]$dtype)
    }

    unflatten(out_tree, leaves)
  })

  environment(wrapper) <- wrapper_env
  wrapper
}
