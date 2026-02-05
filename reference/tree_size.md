# Tree Size

Counts the number of leaf nodes in a tree. This equals the length of the
flat list produced by
[`flatten()`](https://r-xla.github.io/anvil/reference/flatten.md) on the
original structure.

## Usage

``` r
tree_size(x)
```

## Arguments

- x:

  (`Node`)  
  A tree node as returned by
  [`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md).

## Value

A scalar `integer`.

## See also

[`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md),
[`flatten()`](https://r-xla.github.io/anvil/reference/flatten.md)

## Examples

``` r
tree <- build_tree(list(a = 1, b = list(c = 2, d = 3)))
tree_size(tree)
#> [1] 3

tree_size(build_tree(list(1)))
#> [1] 1
```
