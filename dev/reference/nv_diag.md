# Diagonal Matrix

Creates a diagonal matrix from a 1-D tensor.

## Usage

``` r
nv_diag(x)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md))  
  A 1-D tensor of length `n` whose elements become the diagonal entries.

## Value

[`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md)  
An `n x n` matrix with `x` on the diagonal and zeros elsewhere.

## Examples

``` r
jit_eval({
  nv_diag(nv_tensor(c(1, 2, 3)))
})
#> AnvilTensor
#>  1 0 0
#>  0 2 0
#>  0 0 3
#> [ CPUf32{3,3} ] 
```
