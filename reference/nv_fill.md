# Fill Constant

Creates a tensor filled with a scalar value. More memory-efficient than
`nv_tensor(value, shape = shape)` for large tensors.

## Usage

``` r
nv_fill(value, shape, dtype = NULL, ambiguous = FALSE)
```

## Arguments

- value:

  (`numeric(1)`)  
  Scalar value to fill the tensor with.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the output tensor.

- dtype:

  (`character(1)` \| `NULL`)  
  Data type. If `NULL` (default), inferred from `value`.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for more details.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the given `shape` and `dtype`.

## See also

[`nvl_fill()`](https://r-xla.github.io/anvil/reference/nvl_fill.md) for
the underlying primitive.

## Examples

``` r
jit_eval(nv_fill(0, shape = c(2, 3)))
#> AnvilTensor
#>  0 0 0
#>  0 0 0
#> [ CPUf32{2,3} ] 
```
