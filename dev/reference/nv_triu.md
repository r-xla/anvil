# Upper Triangular Matrix

Returns the upper triangular part of a 2-D array, setting elements below
the specified diagonal to zero.

## Usage

``` r
nv_triu(operand, diagonal = 0L)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

- diagonal:

  (`integer(1)`)  
  Diagonal offset. `0` (default) is the main diagonal, positive values
  exclude diagonals above, negative values include diagonals below.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as `operand`.

## See also

[`nv_tril()`](https://r-xla.github.io/anvil/dev/reference/nv_tril.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_fill(1, c(3, 3))
  nv_triu(x)
})
}
```
