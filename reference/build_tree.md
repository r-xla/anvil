# Build Tree

Captures the nesting structure of an object as a tree of `Node`s. Each
leaf in the input becomes a `LeafNode` with an integer index
corresponding to its position in the flat list produced by
[`flatten()`](https://r-xla.github.io/anvil/reference/flatten.md). Lists
become `ListNode`s that record child nodes and names. The resulting tree
can be passed to
[`unflatten()`](https://r-xla.github.io/anvil/reference/unflatten.md) to
reconstruct the original structure from a flat list.

## Usage

``` r
build_tree(x, counter = NULL)
```

## Arguments

- x:

  (any)  
  Object whose structure to capture. Lists are recursed into; everything
  else is a leaf.

- counter:

  (NULL \| environment)  
  Internal counter for assigning leaf indices. Mostly used internally
  and otherwise left as `NULL` (default.)

## Value

A `Node` (`LeafNode` for scalars, `ListNode` for lists).

## See also

[`flatten()`](https://r-xla.github.io/anvil/reference/flatten.md),
[`unflatten()`](https://r-xla.github.io/anvil/reference/unflatten.md),
[`tree_size()`](https://r-xla.github.io/anvil/reference/tree_size.md),
[`reindex_tree()`](https://r-xla.github.io/anvil/reference/reindex_tree.md)

## Examples

``` r
x <- list(a = 1, b = list(c = 2, d = 3))
tree <- build_tree(x)
tree_size(tree)
#> [1] 3

flat <- flatten(x)
unflatten(tree, flat)
#> $a
#> [1] 1
#> 
#> $b
#> $b$c
#> [1] 2
#> 
#> $b$d
#> [1] 3
#> 
#> 
```
