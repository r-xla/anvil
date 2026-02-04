# Primitive If

Conditional execution of branches.

## Usage

``` r
nvl_if(pred, true, false)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of boolean type, scalar)  
  Predicate.

- true:

  (`expression`)  
  Expression for true branch.

- false:

  (`expression`)  
  Expression for false branch.

## Value

Result of the executed branch.

## Shapes

`pred` must be scalar. Both branches must return outputs with the same
shapes.

## StableHLO

Calls
[`stablehlo::hlo_if()`](https://r-xla.github.io/stablehlo/reference/hlo_if.html).

## Examples

``` r
jit_eval(nvl_if(nv_scalar(TRUE), nv_scalar(1), nv_scalar(2)))
#> AnvilTensor
#>  1
#> [ CPUf32{} ] 
```
