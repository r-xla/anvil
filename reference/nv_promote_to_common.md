# Promote Tensors to a Common Dtype

Promote tensors to a common data type, see
[`common_dtype`](https://r-xla.github.io/anvil/reference/common_dtype.md)
for more details.

## Usage

``` r
nv_promote_to_common(...)
```

## Arguments

- ...:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensors to promote.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))

## Examples

``` r
jit_eval({
  x <- nv_tensor(1L)
  y <- nv_tensor(1.5)
  # integer is promoted to float
  nv_promote_to_common(x, y)
})
#> [[1]]
#> AnvilTensor
#>  1
#> [ CPUf32{1} ] 
#> 
#> [[2]]
#> AnvilTensor
#>  1.5000
#> [ CPUf32{1} ] 
#> 
```
