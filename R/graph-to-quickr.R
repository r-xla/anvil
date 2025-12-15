#' @include graph-to-quickr-r.R
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
#' Currently supported primitives are:
#' `constant`, `add`, `sub`, `mul`, `divide`, `negate`, `broadcast_in_dim`,
#' `dot_general`, `transpose`, `reshape`, `sum`.
#' The code generator currently supports tensors up to rank 5. Some primitives
#' are more restricted (e.g. `transpose` currently only handles rank-2 tensors).
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
  needs_wrapper <- isTRUE(needs_pack) || length(graph@constants)

  r_fun <- graph_to_quickr_r_function(graph, include_declare = TRUE, pack_output = needs_pack)
  inner_quick <- quickr_eager_compile(r_fun)

  if (!isTRUE(needs_wrapper)) {
    return(inner_quick)
  }

  r_arg_names <- names(formals(r_fun))
  n_user <- length(graph@inputs)
  user_arg_names <- r_arg_names[seq_len(n_user)]

  const_args <- list()
  if (length(graph@constants)) {
    const_arg_names <- r_arg_names[(n_user + 1L):(n_user + length(graph@constants))]
    const_vals <- lapply(graph@constants, function(node) {
      as_array(node@aval@data)
    })
    const_args <- stats::setNames(const_vals, const_arg_names)
  }

  if (!isTRUE(needs_pack)) {
    wrapper <- function() {
      stop("internal placeholder")
    }
    formals(wrapper) <- formals(r_fun)[seq_len(n_user)]

    wrapper_env <- new.env(parent = environment())
    wrapper_env$inner_quick <- inner_quick
    wrapper_env$user_arg_names <- user_arg_names
    wrapper_env$const_args <- const_args

    body(wrapper) <- quote({
      args <- mget(user_arg_names, envir = environment(), inherits = FALSE)
      do.call(inner_quick, c(const_args, args))
    })

    environment(wrapper) <- wrapper_env
    return(wrapper)
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

  wrapper <- function() {
    stop("internal placeholder")
  }
  formals(wrapper) <- formals(r_fun)[seq_len(n_user)]

  wrapper_env <- new.env(parent = environment())
  wrapper_env$inner_quick <- inner_quick
  wrapper_env$out_tree <- graph@out_tree
  wrapper_env$out_infos <- out_infos
  wrapper_env$out_lens <- out_lens
  wrapper_env$user_arg_names <- user_arg_names
  wrapper_env$const_args <- const_args

  body(wrapper) <- quote({
    args <- mget(user_arg_names, envir = environment(), inherits = FALSE)
    packed <- do.call(inner_quick, c(const_args, args))

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
        return(array(base, dim = shape))
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
