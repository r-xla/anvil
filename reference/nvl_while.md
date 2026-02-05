# Primitive While Loop

Repeatedly executes `body` while `cond` returns `TRUE`, like R's `while`
loop. The loop state is initialized with `init` and passed through each
iteration. Otherwise, no state is maintained between iterations.

## Usage

``` r
nvl_while(init, cond, body)
```

## Arguments

- init:

  (`named list()`)  
  Named list of initial state values.

- cond:

  (`function`)  
  Condition function that receives the current state as arguments and
  outputs whether to continue the loop.

- body:

  (`function`)  
  Body function that receives the current state as arguments and returns
  a named list with the same structure, dtypes, and shapes as `init`.

## Value

Named list with the same structure as `init` containing the final state
after the loop terminates.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`stablehlo::hlo_while()`](https://r-xla.github.io/stablehlo/reference/hlo_while.html).

## See also

[`nv_while()`](https://r-xla.github.io/anvil/reference/nv_while.md)

## Examples

``` r
jit_eval({
  nvl_while(
    init = list(i = 0L, total = 0L),
    cond = function(i, total) i <= 5L,
    body = function(i, total) list(
      i = i + 1L,
      total = total + i
    )
  )
})
#> $i
#> AnvilTensor
#>  6
#> [ CPUi32?{} ] 
#> 
#> $total
#> AnvilTensor
#>  15
#> [ CPUi32?{} ] 
#> 
```
