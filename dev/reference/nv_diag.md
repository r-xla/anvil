# Diagonal Matrix

Creates a diagonal matrix from a 1-D array.

## Usage

``` r
nv_diag(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  A 1-D array of length `n` whose elements become the diagonal entries.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
An `n x n` matrix with `x` on the diagonal and zeros elsewhere.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  nv_diag(nv_array(c(1, 2, 3)))
})
}
```
