# Flatten

Recursively flattens a nested list into a single flat list containing
only the leaf values, preserving left-to-right order.

Currently only lists are flattened and all other objects are treated as
leaves.

Use
[`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md)
to capture the nesting structure so it can be restored with
[`unflatten()`](https://r-xla.github.io/anvil/reference/unflatten.md).

## Usage

``` r
flatten(x)
```

## Arguments

- x:

  (any)  
  Object to flatten.

## Value

[`list()`](https://rdrr.io/r/base/list.html) containing the flattened
values.

## See also

[`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md),
[`unflatten()`](https://r-xla.github.io/anvil/reference/unflatten.md),
[`tree_size()`](https://r-xla.github.io/anvil/reference/tree_size.md)

## Examples

``` r
x <- list(a = 1, b = list(c = 2, d = 3))
flatten(x)
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2
#> 
#> [[3]]
#> [1] 3
#> 

flatten(list(1:3, "hello"))
#> [[1]]
#> [1] 1 2 3
#> 
#> [[2]]
#> [1] "hello"
#> 
```
