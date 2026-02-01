# Pack and unpack
flatten_fun <- function(f, ..., in_node = NULL) {
  if (is.null(in_node)) {
    in_node <- build_tree(list(...))
  } else if (...length()) {
    cli_abort("in_node is not compatible with ... arguments")
  }
  f_orig <- f
  f <- function(...) {
    # We could do this out of the function and re-use,
    # but because we always jit, we don't worry about it
    # (at least for now)
    args <- unflatten(in_node, list(...))

    outs <- do.call(f_orig, args)
    list(
      build_tree(outs),
      flatten(outs)
    )
  }
  class(f) <- "FlattenedFunction"
  f
}

new_counter <- function() {
  y <- hashtab(size = 1L)
  y[["i"]] <- 0L
  y
}

#' @title Flatten
#' @description
#' Flatten a nested structure into a flat list.
#' @param x Object to flatten.
#' @return A flat list.
#' @export
flatten <- function(x) {
  UseMethod("flatten")
}

#' @export
flatten.list <- function(x) {
  out <- lapply(unname(x), flatten)
  Reduce(c, out)
}

#' @export
flatten.default <- function(x) {
  list(x)
}

#' @title Build Tree
#' @description
#' Build a tree structure from a nested object for tracking structure during flattening/unflattening.
#' @param x Object to build tree from.
#' @param counter Internal counter for leaf indices.
#' @return `Node`
#' @export
build_tree <- function(x, counter = NULL) {
  UseMethod("build_tree")
}

#' @export
build_tree.list <- function(x, counter = NULL) {
  counter <- counter %??% new_counter()
  out <- lapply(unname(x), build_tree, counter = counter)
  # basically non-recursive unlist that maintains list structure even for atomics (unlist(list(1, 2))
  # would become c(1, 2))
  ListNode(
    out,
    names(x)
  )
}

#' @export
build_tree.default <- function(x, counter = NULL) {
  counter <- counter %??% new_counter()
  i <- counter[["i"]] + 1L
  counter[["i"]] <- i
  LeafNode(i)
}

mark_some <- function(x, marked) {
  stopifnot(is.list(x))
  structure(list(data = x, marked = marked), class = "MarkedArgs")
}

#' @export
build_tree.MarkedArgs <- function(x, counter = NULL) {
  if (is.null(counter)) {
    counter <- new_counter()
  }

  n <- length(x$data)
  subsize <- vector("integer", n)
  nodes <- vector("list", n)
  prev <- 0L
  for (i in seq_along(x$data)) {
    nodes[[i]] <- build_tree(x$data[[i]], counter = counter)
    subsize[i] <- counter[["i"]] - prev
    prev <- prev + subsize[i]
  }

  if (!is.null(x$marked)) {
    is_marked <- rlang::names2(x$data) %in% x$marked
    is_marked_flat <- rep(is_marked, times = subsize)
  } else {
    cli_abort("Internal error: marked must be non-NULL")
  }

  MarkedListNode(nodes, names(x$data), is_marked_flat)
}


#' @title Unflatten
#' @description
#' Reconstruct a nested structure from a flat list using a tree structure.
#' @param node Tree node describing the structure.
#' @param x Flat list to unflatten.
#' @return Reconstructed nested structure.
#' @export
unflatten <- function(node, x) {
  UseMethod("unflatten")
}

#' @export
unflatten.LeafNode <- function(node, x) {
  x[[node$i]]
}

#' @export
unflatten.ListNode <- function(node, x) {
  stats::setNames(lapply(node$nodes, unflatten, x = x), node$names)
}

LeafNode <- function(i) {
  structure(list(i = i), class = c("LeafNode", "Node"))
}

ListNode <- function(nodes, names) {
  structure(
    list(
      nodes = unname(nodes),
      names = names
    ),
    class = c("ListNode", "Node")
  )
}

MarkedListNode <- function(nodes, names, marked) {
  structure(
    list(
      nodes = unname(nodes),
      names = names,
      marked = marked
    ),
    class = c("MarkedListNode", "ListNode", "Node")
  )
}

#' @title Tree Size
#' @description
#' Get the number of leaf nodes in a tree.
#' @param x Tree node.
#' @return Integer count of leaf nodes.
#' @export
tree_size <- function(x) {
  UseMethod("tree_size")
}

#' @export
tree_size.LeafNode <- function(x) {
  1L
}

#' @export
tree_size.ListNode <- function(x) {
  sum(vapply(x$nodes, tree_size, integer(1L)))
}

#' @export
tree_size.MarkedListNode <- function(x) {
  sum(vapply(x$nodes, tree_size, integer(1L)))
}

filter_list_node <- function(tree, names) {
  stopifnot(inherits(tree, "ListNode"))
  if (is.null(tree$names)) {
    cli_abort("tree must have names")
  }
  keep_idx <- which(tree$names %in% names)
  if (length(keep_idx) == length(tree$names)) {
    return(tree)
  }
  counter <- new_counter()
  renumbered_nodes <- lapply(tree$nodes[keep_idx], reindex_tree, counter = counter)
  ListNode(renumbered_nodes, tree$names[keep_idx])
}

#' @title Reindex Tree
#' @description
#' Recursively reindex leaf nodes starting from a counter.
#' @param x Tree node to reindex.
#' @param counter Counter object for generating new indices.
#' @return Reindexed tree node.
#' @export
reindex_tree <- function(x, counter) {
  UseMethod("reindex_tree")
}

#' @export
reindex_tree.LeafNode <- function(x, counter) {
  i <- counter[["i"]] + 1L
  counter[["i"]] <- i
  LeafNode(i)
}

#' @export
reindex_tree.ListNode <- function(x, counter) {
  reindexed <- lapply(x$nodes, reindex_tree, counter = counter)
  ListNode(reindexed, x$names)
}

flat_mask_from_names <- function(tree, names) {
  if (is.null(names) || length(names) == 0L) {
    rep(TRUE, times = tree_size(tree))
  } else {
    mask <- tree$names %in% names
    rep(mask, times = vapply(tree$nodes, tree_size, integer(1L)))
  }
}
