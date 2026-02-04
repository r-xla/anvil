# Primitive While Loop

Executes a while loop.

## Usage

``` r
nvl_while(init, cond, body)
```

## Arguments

- init:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Named list of initial state values.

- cond:

  (`function`)  
  Condition function returning boolean.

- body:

  (`function`)  
  Body function returning updated state.

## Value

Final state after loop terminates.

## Shapes

`cond` must return a scalar boolean. `body` must return the same shapes
as `init`.

## StableHLO

Calls
[`stablehlo::hlo_while()`](https://r-xla.github.io/stablehlo/reference/hlo_while.html).

## Examples

``` r
jit_eval({
  nvl_while(
    init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
    cond = function(i, total) nvl_lt(i, nv_scalar(5L)),
    body = function(i, total) list(
      i = nvl_add(i, nv_scalar(1L)),
      total = nvl_add(total, i)
    )
  )
})
#> $i
#> AnvilTensor
#>  5
#> [ CPUi32{} ] 
#> 
#> $total
#> AnvilTensor
#>  10
#> [ CPUi32{} ] 
#> 
```
