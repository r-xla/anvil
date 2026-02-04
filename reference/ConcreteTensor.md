# Concrete Tensor Class

An
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
that also holds a reference to the actual tensor data. Usually
represents a closed-over constant in a program. Inherits from
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md).

## Usage

``` r
ConcreteTensor(data)
```

## Arguments

- data:

  ([`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md))  
  The actual tensor data.

## Lowering

When lowering to XLA, these become inputs to the executable instead of
embedding them into programs as constants. This is to avoid increasing
compilation time and bloating the size of the executable.

## Examples

``` r
y <- nv_tensor(c(0.5, 0.6))
x <- ConcreteTensor(y)
x
#> ConcreteTensor
#>  0.5000
#>  0.6000
#> [ CPUf32{2} ] 
ambiguous(x)
#> [1] FALSE
shape(x)
#> [1] 2
ndims(x)
#> [1] 1
dtype(x)
#> <f32>

# How it appears during tracing
graph <- trace_fn(function() y, list())
graph
#> <AnvilGraph>
#>   Inputs: (none)
#>   Constants:
#>     %c1: f32[2]
#>   Body: (empty)
#>   Outputs:
#>     %c1: f32[2] 
graph$outputs[[1]]$aval
#> ConcreteTensor
#>  0.5000
#>  0.6000
#> [ CPUf32{2} ] 
```
