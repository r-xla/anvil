# Unflatten

Reconstructs a nested structure from a flat list by using a tree
previously created with
[`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md).
Each `LeafNode` in the tree selects the corresponding element from `x`
by index, and `ListNode`s restore the original nesting and names.

## Usage

``` r
unflatten(node, x)
```

## Arguments

- node:

  (`Node`)  
  Tree describing the target structure, as returned by
  [`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md).

- x:

  (list)  
  Flat list of leaf values, typically produced by
  [`flatten()`](https://r-xla.github.io/anvil/reference/flatten.md).

## Value

The reconstructed nested structure (list or single value).

## See also

[`flatten()`](https://r-xla.github.io/anvil/reference/flatten.md),
[`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md)

## Examples

``` r
x <- list(a = 1, b = list(c = 2, d = 3))
tree <- build_tree(x)
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

unflatten(tree, list(10, 20, 30))
#> $a
#> [1] 10
#> 
#> $b
#> $b$c
#> [1] 20
#> 
#> $b$d
#> [1] 30
#> 
#> 
```
