# Primitive If

Conditional execution of one of two branches based on a scalar boolean
predicate. Unlike
[`nvl_ifelse()`](https://r-xla.github.io/anvil/reference/nvl_ifelse.md)
which operates element-wise, this evaluates only the selected branch.

## Usage

``` r
nvl_if(pred, true, false)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Scalar boolean predicate that determines which branch to execute.

- true, false:

  (NSE)  
  Expressions for the true and false branches. Both must return outputs
  with the same structure, dtypes, and shapes.

## Value

Result of the executed branch.  
An output is ambiguous if it is ambiguous in both branches.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_if()`](https://r-xla.github.io/stablehlo/reference/hlo_if.html).

## See also

[`nv_if()`](https://r-xla.github.io/anvil/reference/nv_if.md),
[`nvl_ifelse()`](https://r-xla.github.io/anvil/reference/nvl_ifelse.md)

## Examples

``` r
jit_eval(nvl_if(nv_scalar(TRUE), nv_scalar(1), nv_scalar(2)))
#> AnvilTensor
#>  1
#> [ CPUf32{} ] 
```
