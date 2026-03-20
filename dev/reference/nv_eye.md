# Identity Matrix

Creates an `n x n` identity matrix.

## Usage

``` r
nv_eye(n, dtype = "f32")
```

## Arguments

- n:

  (`integer(1)`)  
  Size of the identity matrix.

- dtype:

  (`character(1)` \|
  [`tengen::TensorDataType`](https://r-xla.github.io/tengen/reference/TensorDataType.html))  
  Data type.

## Value

[`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md)  
An `n x n` identity matrix.

## See also

[`nv_diag()`](https://r-xla.github.io/anvil/dev/reference/nv_diag.md)
for general diagonal matrices.

## Examples

``` r
jit_eval(nv_eye(3L))
#> AnvilTensor
#>  1 0 0
#>  0 1 0
#>  0 0 1
#> [ CPUf32{3,3} ] 
```
