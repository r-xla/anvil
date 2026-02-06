#' @include graph-to-quickr-r.R
NULL

#' Convert an AnvilGraph to a quickr-compiled function
#'
#' Lowers a supported subset of `AnvilGraph` objects to a plain R function and
#' compiles it with `quickr::quick()`.
#'
#' The returned function expects plain R scalars/vectors/arrays (not
#' [`AnvilTensor`]) and returns plain R values/arrays.
#'
#' If the graph returns multiple outputs (e.g. a nested list), the compiled
#' function returns the same structure by packing/unpacking values for `quickr`.
#'
#' Currently supported primitives are:
#' `fill`, `convert`, `add`, `sub`, `mul`, `divide`, `negate`, `abs`, `sqrt`,
#' `log`, `exp`, `floor`, `ceil`, `power`, `maximum`, `minimum`, `equal`,
#' `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, `and`, `or`,
#' `xor`, `not`, `select`, `broadcast_in_dim`, `dot_general`, `transpose`,
#' `reshape`, `sum`, `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`.
#' The code generator currently supports tensors up to rank 5. Some primitives
#' are more restricted (e.g. `transpose` currently only handles rank-2 tensors).
#'
#' @param graph ([`AnvilGraph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @export
graph_to_quickr_function <- function(graph) {
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls AnvilGraph}")
  }

  assert_quickr_installed("{.fn graph_to_quickr_function}")

  in_tree <- graph$in_tree
  needs_flatten <- inherits(in_tree, "ListNode") && any(vapply(in_tree$nodes, inherits, logical(1L), "ListNode"))

  needs_pack <- !(inherits(graph$out_tree, "LeafNode") && length(graph$outputs) == 1L)
  out_infos <- lapply(graph$outputs, function(node) {
    list(dtype = as.character(dtype(node)), shape = shape(node))
  })
  needs_dimfix <- !isTRUE(needs_pack) &&
    length(out_infos) == 1L &&
    length(out_infos[[1L]]$shape) == 1L

  needs_wrapper <- isTRUE(needs_pack) || length(graph$constants) || isTRUE(needs_flatten) || isTRUE(needs_dimfix)

  r_fun <- graph_to_quickr_r_function(graph, include_declare = TRUE, pack_output = needs_pack)
  inner_quick <- quickr_eager_compile(r_fun)

  if (!isTRUE(needs_wrapper)) {
    return(inner_quick)
  }

  r_arg_names <- names(formals(r_fun)) %||% character()
  n_user <- length(graph$inputs)
  leaf_arg_names <- r_arg_names[seq_len(n_user)]

  const_args <- list()
  if (length(graph$constants)) {
    const_arg_names <- r_arg_names[(n_user + 1L):(n_user + length(graph$constants))]
    const_vals <- lapply(graph$constants, function(node) {
      as_array(node$aval$data)
    })
    const_args <- stats::setNames(const_vals, const_arg_names)
  }

  out_lens <- vapply(
    out_infos,
    function(info) {
      if (!length(info$shape)) 1L else Reduce(`*`, as.integer(info$shape), init = 1L)
    },
    integer(1L)
  )

  wrapper <- function() {}
  if (isTRUE(needs_flatten)) {
    top_names <- in_tree$names %||% rep("", length(in_tree$nodes))
    top_names <- vapply(
      top_names,
      function(x) {
        if (is.null(x) || !nzchar(x)) "x" else x
      },
      character(1L)
    )
    top_names <- make.unique(make.names(top_names))
    formals(wrapper) <- as.pairlist(stats::setNames(rep(list(quote(expr = )), length(top_names)), top_names))
  } else {
    formals(wrapper) <- formals(r_fun)[seq_len(n_user)]
  }

  wrapper_env <- new.env(parent = environment())
  wrapper_env$inner_quick <- inner_quick
  wrapper_env$out_tree <- graph$out_tree
  wrapper_env$out_infos <- out_infos
  wrapper_env$out_lens <- out_lens
  wrapper_env$needs_dimfix <- needs_dimfix
  wrapper_env$leaf_arg_names <- leaf_arg_names
  wrapper_env$needs_flatten <- needs_flatten
  wrapper_env$top_names <- if (isTRUE(needs_flatten)) top_names else NULL
  wrapper_env$const_args <- const_args

  body(wrapper) <- quote({
    if (isTRUE(needs_flatten)) {
      args_top <- mget(top_names, envir = environment(), inherits = FALSE)
      args <- flatten(args_top)
    } else if (!length(leaf_arg_names)) {
      args <- list()
    } else {
      args <- mget(leaf_arg_names, envir = environment(), inherits = FALSE)
    }

    if (length(args) != length(leaf_arg_names)) {
      cli_abort("Expected {length(leaf_arg_names)} inputs, got {length(args)}")
    }

    args <- stats::setNames(args, leaf_arg_names)
    packed <- do.call(inner_quick, c(const_args, args))

    if (!isTRUE(needs_pack)) {
      if (isTRUE(needs_dimfix)) {
        return(array(packed, dim = as.integer(out_infos[[1L]]$shape)))
      }
      return(packed)
    }

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
