# Iota Tensor Class

An
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
representing an integer sequence. Usually created by
[`nv_iota()`](https://r-xla.github.io/anvil/reference/nv_iota.md) /
[`nv_seq()`](https://r-xla.github.io/anvil/reference/nv_seq.md), which
both call
[`nvl_iota()`](https://r-xla.github.io/anvil/reference/nvl_iota.md)
internally. Inherits from
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md).

## Usage

``` r
IotaTensor(shape, dtype, dimension, start = 1L, ambiguous = FALSE)
```

## Arguments

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the tensor.

- dtype:

  ([`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  The data type.

- dimension:

  (`integer(1)`)  
  The dimension along which values increase.

- start:

  (`integer(1)`)  
  The starting value.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for more details.

## Lowering

When lowering to stableHLO, these become `iota` operations that generate
the integer sequence so they do not need to actually hold the data in
the executable, similar to `ALTREP`s in R. It lowers to
[`stablehlo::hlo_iota()`](https://r-xla.github.io/stablehlo/reference/hlo_iota.html),
optionally shifting the starting value via
[`stablehlo::hlo_add()`](https://r-xla.github.io/stablehlo/reference/hlo_add.html).

## Examples

``` r
x <- IotaTensor(shape = 4L, dtype = "i32", dimension = 1L)
x
#> IotaTensor(shape=(4), dtype=i32, dimension=1, start=1) 
ambiguous(x)
#> [1] FALSE
shape(x)
#> [1] 4
ndims(x)
#> [1] 1
dtype(x)
#> <i32>
# How it appears during tracing:
graph <- trace_fn(function() nv_iota(dim = 1L, dtype = "i32", shape = 4L), list())
graph
#> <AnvilGraph>
#>   Inputs: (none)
#>   Body:
#>     %1: i32[4] = iota [dim = 1, dtype = i32, shape = 4, start = 1, ambiguous = FALSE] ()
#>   Outputs:
#>     %1: i32[4] 
graph$outputs[[1]]$aval
#> IotaTensor(shape=(4), dtype=i32, dimension=1, start=1) 
```
