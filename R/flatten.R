# Pack and unpack
flatten_fun <- function(f, ...) {
  # (flat_args) -> (flat_out, out_node)
  in_node <- build_tree(list(...))
  function(...) {
    # We could do this out of the function and re-use,
    # but because we always jit, we don't worry about it
    # (at least for now)
    args <- unflatten(in_node, list(...))

    outs <- do.call(f, args)
    list(
      build_tree(outs),
      flatten(outs)
    )
  }
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
