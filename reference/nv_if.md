# Conditional Branching

Conditional execution of two branches. Unlike
[`nv_ifelse()`](https://r-xla.github.io/anvil/reference/nv_ifelse.md),
which selects elements, this executes only one of the two branches
depending on a scalar predicate.

## Usage

``` r
nv_if(pred, true, false)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of boolean type, scalar)  
  Predicate.

- true:

  (`expression`)  
  Expression for the true branch (non-standard evaluation).

- false:

  (`expression`)  
  Expression for the false branch (non-standard evaluation). Must return
  outputs with the same shapes as the true branch.

## Value

Result of the executed branch.

## See also

[`nvl_if()`](https://r-xla.github.io/anvil/reference/nvl_if.md) for the
underlying primitive,
[`nv_ifelse()`](https://r-xla.github.io/anvil/reference/nv_ifelse.md)
for element-wise selection.

## Examples

``` r
jit_eval(nv_if(nv_scalar(TRUE), nv_scalar(1), nv_scalar(2)))
#> AnvilTensor
#>  1
#> [ CPUf32{} ] 
```
