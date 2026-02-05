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
#' Recursively flattens a nested list into a single flat list containing only the
#' leaf values, preserving left-to-right order.
#'
#' Currently only lists are flattened and all other objects are treated as leaves.
#'
#' Use [build_tree()] to capture the nesting structure so it can be restored with
#' [unflatten()].
#' @param x (any)\cr
#'   Object to flatten.
#' @return `list()` containing the flattened values.
#' @seealso [build_tree()], [unflatten()], [tree_size()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' flatten(x)
#'
#' flatten(list(1:3, "hello"))
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
#' Captures the nesting structure of an object as a tree of `Node`s. Each leaf
#' in the input becomes a `LeafNode` with an integer index corresponding to its
#' position in the flat list produced by [flatten()]. Lists become `ListNode`s
#' that record child nodes and names. The resulting tree can be passed to
#' [unflatten()] to reconstruct the original structure from a flat list.
#' @param x (any)\cr
#'   Object whose structure to capture. Lists are recursed into; everything else
#'   is a leaf.
#' @param counter (NULL | environment)\cr
#'   Internal counter for assigning leaf indices. Mostly used internally and otherwise left as
#'   `NULL` (default.)
#' @return A `Node` (`LeafNode` for scalars, `ListNode` for lists).
#' @seealso [flatten()], [unflatten()], [tree_size()], [reindex_tree()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#' tree_size(tree)
#'
#' flat <- flatten(x)
#' unflatten(tree, flat)
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
#' Reconstructs a nested structure from a flat list by using a tree previously
#' created with [build_tree()]. Each `LeafNode` in the tree selects the
#' corresponding element from `x` by index, and `ListNode`s restore the
#' original nesting and names.
#' @param node (`Node`)\cr
#'   Tree describing the target structure, as returned by [build_tree()].
#' @param x (list)\cr
#'   Flat list of leaf values, typically produced by [flatten()].
#' @return The reconstructed nested structure (list or single value).
#' @seealso [flatten()], [build_tree()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#' flat <- flatten(x)
#'
#' unflatten(tree, flat)
#'
#' unflatten(tree, list(10, 20, 30))
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
#' Counts the number of leaf nodes in a tree. This equals the length of the
#' flat list produced by [flatten()] on the original structure.
#' @param x (`Node`)\cr
#'   A tree node as returned by [build_tree()].
#' @return A scalar `integer`.
#' @seealso [build_tree()], [flatten()]
#' @examples
#' tree <- build_tree(list(a = 1, b = list(c = 2, d = 3)))
#' tree_size(tree)
#'
#' tree_size(build_tree(list(1)))
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

#' @title Filter List Node
#' @description
#' Subsets a `ListNode` to keep only the children whose names match `names`,
#' then reindexes the leaf nodes so they map to contiguous positions in a flat
#' list. If all names are kept the original tree is returned unchanged.
#' @param tree (`ListNode`)\cr
#'   A named list node as returned by [build_tree()].
#' @param names (character)\cr
#'   Names of children to keep.
#' @return A `ListNode` containing only the selected children with reindexed
#'   leaves.
#' @seealso [build_tree()], [reindex_tree()], [unflatten()]
#' @examples
#' x <- list(a = 1, b = 2, c = 3)
#' tree <- build_tree(x)
#' sub <- filter_list_node(tree, c("a", "c"))
#' tree_size(sub)
#'
#' unflatten(sub, x[c("a", "c")])
#' @export
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
#' Reassigns leaf indices so they form a contiguous sequence starting from the
#' current counter value. This is used internally after filtering nodes from a tree (e.g.
#' via [filter_list_node()]) to ensure leaf indices still map correctly to
#' positions in a flat list.
#' Not intended for direct use.
#' @param x (`Node`)\cr
#'   A tree node to reindex.
#' @param counter (environment)\cr
#'   A mutable counter created by `new_counter()`.
#' @return A new `Node` with updated leaf indices.
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
