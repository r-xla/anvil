# Filter List Node

Subsets a `ListNode` to keep only the children whose names match
`names`, then reindexes the leaf nodes so they map to contiguous
positions in a flat list. If all names are kept the original tree is
returned unchanged.

## Usage

``` r
filter_list_node(tree, names)
```

## Arguments

- tree:

  (`ListNode`)  
  A named list node as returned by
  [`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md).

- names:

  (character)  
  Names of children to keep.

## Value

A `ListNode` containing only the selected children with reindexed
leaves.

## See also

[`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md),
[`reindex_tree()`](https://r-xla.github.io/anvil/reference/reindex_tree.md),
[`unflatten()`](https://r-xla.github.io/anvil/reference/unflatten.md)

## Examples

``` r
x <- list(a = 1, b = 2, c = 3)
tree <- build_tree(x)
sub <- filter_list_node(tree, c("a", "c"))
tree_size(sub)
#> [1] 2

unflatten(sub, x[c("a", "c")])
#> $a
#> [1] 1
#> 
#> $c
#> [1] 3
#> 
```
