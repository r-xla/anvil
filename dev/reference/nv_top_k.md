# Top-K Elements

Returns the `k` largest values along a dimension, sorted in decreasing
order.

## Usage

``` r
nv_top_k(x, k, dim = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array.

- k:

  (`integer(1)`)  
  Number of top elements to return. Must satisfy
  `1 <= k <= shape(x)[dim]`.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to take the top `k`. If `NULL` (default), uses
  the last dimension.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Same shape as `x` except `dim` has size `k`. Values are sorted in
decreasing order along `dim`.

## See also

[`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md)
for the underlying primitive,
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md).

## Examples

``` r
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
nv_top_k(x, k = 3L)
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 

m <- nv_array(matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE))
nv_top_k(m, k = 2L, dim = 2L)
#> AnvlArray
#>  5 3
#>  4 2
#> [ CPUf32{2,2} ] 
```
