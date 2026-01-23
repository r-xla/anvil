# Tensor-like Objects

A value that is either an
[`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md),
can be converted to it, or represents an abstract version of it. This
also includes atomic R vectors.

## See also

[nv_tensor](https://r-xla.github.io/anvil/reference/AnvilTensor.md),
[ConcreteTensor](https://r-xla.github.io/anvil/reference/ConcreteTensor.md),
[AbstractTensor](https://r-xla.github.io/anvil/reference/AbstractTensor.md),
[LiteralTensor](https://r-xla.github.io/anvil/reference/LiteralTensor.md),
[GraphBox](https://r-xla.github.io/anvil/reference/GraphBox.md)

## Examples

``` r
x <- nv_tensor(1:4, dtype = "f32")
x
#> AnvilTensor
#>  1
#>  2
#>  3
#>  4
#> [ CPUf32{4} ] 
```
