#' @include graph-to-r.R
NULL

#' Convert a Graph to a quickr-compiled function
#'
#' Lowers a supported subset of `anvil::Graph` objects to a plain R function and
#' compiles it with `quickr::quick()`.
#'
#' If the graph returns multiple outputs (e.g. a nested list), the compiled
#' function returns the same structure by packing/unpacking values for {quickr}.
#'
#' At the moment this only supports graphs with a flat (non-nested) argument
#' list.
#'
#' @param graph ([`Graph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @export
graph_to_quickr_function <- function(graph) {
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls anvil::Graph}")
  }

  assert_quickr_installed("{.fn graph_to_quickr_function}")

  in_tree <- graph@in_tree
  if (inherits(in_tree, "ListNode") && any(vapply(in_tree$nodes, inherits, logical(1L), "ListNode"))) {
    cli_abort(c(
      "{.fn graph_to_quickr_function} currently supports only flat (non-nested) argument lists.",
      i = "Pass tensors as top-level arguments, not nested lists."
    ))
  }

  needs_pack <- !(inherits(graph@out_tree, "LeafNode") && length(graph@outputs) == 1L)

  r_fun <- graph_to_quickr_r_function(graph, include_declare = TRUE, pack_output = needs_pack)
  inner_quick <- quickr_eager_compile(r_fun)

  if (!isTRUE(needs_pack)) {
    return(inner_quick)
  }

  out_infos <- lapply(graph@outputs, function(node) {
    if (is_graph_value(node)) {
      list(dtype = as.character(node@aval@dtype), shape = node@aval@shape@dims)
    } else {
      list(dtype = as.character(node@dtype), shape = integer())
    }
  })
  out_lens <- vapply(out_infos, function(info) {
    if (!length(info$shape)) 1L else Reduce(`*`, as.integer(info$shape), init = 1L)
  }, integer(1L))
  arg_names <- names(formals(r_fun))

  wrapper <- function() {
    stop("internal placeholder")
  }
  formals(wrapper) <- formals(r_fun)

  wrapper_env <- new.env(parent = environment())
  wrapper_env$inner_quick <- inner_quick
  wrapper_env$out_tree <- graph@out_tree
  wrapper_env$out_infos <- out_infos
  wrapper_env$out_lens <- out_lens
  wrapper_env$arg_names <- arg_names

  body(wrapper) <- quote({
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    args <- args[arg_names]

    packed <- do.call(inner_quick, args)

    decode_leaf <- function(seg, shape, dtype) {
      base <- if (dtype %in% c("pred", "i1")) {
        seg != 0
      } else if (grepl("^(u?i)(8|16|32|64)$", dtype)) {
        as.integer(seg)
      } else {
        as.double(seg)
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
