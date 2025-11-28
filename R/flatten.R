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
  class(f) <- "anvil::FlattenedFunction"
  f
}

new_counter <- function() {
  y <- hashtab(size = 1L)
  y[["i"]] <- 0L
  y
}

flatten <- S7::new_generic("flatten", "x", function(x) {
  S7::S7_dispatch()
})

method(flatten, S7::new_S3_class("list")) <- function(x) {
  out <- lapply(unname(x), flatten)
  Reduce(c, out)
}

method(flatten, S7::class_any) <- function(x) {
  list(x)
}

build_tree <- S7::new_generic("build_tree", "x", function(x, counter = NULL) {
  counter <- counter %??% new_counter()
  S7::S7_dispatch()
})

method(build_tree, S7::new_S3_class("list")) <- function(x, counter = NULL) {
  out <- lapply(unname(x), build_tree, counter = counter)
  # basically non-recursive unlist that maintains list structure even for atomics (unlist(list(1, 2))
  # would become c(1, 2))
  ListNode(
    out,
    names(x)
  )
}

method(build_tree, S7::class_any) <- function(x, counter = NULL) {
  i <- counter[["i"]] + 1L
  counter[["i"]] <- i
  LeafNode(i)
}


mark_some <- function(x, marked) {
  stopifnot(is.list(x))
  structure(list(data = x, marked = marked), class = "MarkedArgs")
}

method(build_tree, S7::new_S3_class("MarkedArgs")) <- function(x, counter = NULL) {
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
    is_marked <- names(x$data) %in% x$marked
    is_marked_flat <- rep(is_marked, times = subsize)
    # TODO: I think this is wrong???
  } else {
    is_marked_flat <- FALSE
  }

  MarkedListNode(nodes, names(x$data), is_marked_flat)
}


unflatten <- S7::new_generic("unflatten", "node", function(node, x) {
  S7::S7_dispatch()
})

method(unflatten, S7::new_S3_class("LeafNode")) <- function(node, x) {
  x[[node$i]]
}

method(unflatten, S7::new_S3_class("ListNode")) <- function(node, x) {
  stats::setNames(lapply(node$nodes, unflatten, x = x), node$names)
}

#method(unflatten, S7::new_S3_class("NodeList"), function(node, list, i = 1L) {
#  out <- lapply(node$children)
#})
#
#method(unflatten, S7::class_any, function(tree, x) {
#})

#method(unflatten, S7::new_S3_class("NodeHashtab"))
#method(unflatten, S7::new_S3_class("NodeEnvironment"))

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
    class = c("MarkedListNode", "ListNode")
  )
}

tree_size <- new_generic("tree_size", "x", function(x) {
  S7::S7_dispatch()
})

method(tree_size, S7::new_S3_class("LeafNode")) <- function(x) {
  1L
}

method(tree_size, S7::new_S3_class("ListNode")) <- function(x) {
  sum(vapply(x$nodes, tree_size, integer(1L)))
}

method(tree_size, S7::new_S3_class("MarkedListNode")) <- function(x) {
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

# Recursively reindex leaf nodes starting from counter
reindex_tree <- new_generic("reindex_tree", "x", function(x, counter) {
  S7::S7_dispatch()
})

method(reindex_tree, S7::new_S3_class("LeafNode")) <- function(x, counter) {
  i <- counter[["i"]] + 1L
  counter[["i"]] <- i
  LeafNode(i)
}

method(reindex_tree, S7::new_S3_class("ListNode")) <- function(x, counter) {
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
