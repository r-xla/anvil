# While Loop

Executes a functional while loop.

## Usage

``` r
nv_while(init, cond, body)
```

## Arguments

- init:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Named list of initial state values.

- cond:

  (`function`)  
  Condition function returning a scalar boolean. Receives the state
  values as arguments.

- body:

  (`function`)  
  Body function returning the updated state as a named list with the
  same structure as `init`.

## Value

Final state after the loop terminates (same structure as `init`).

## See also

[`nvl_while()`](https://r-xla.github.io/anvil/reference/nvl_while.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  nv_while(
    init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
    cond = function(i, total) i < 5L,
    body = function(i, total) list(
      i = i + 1L,
      total = total + i
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
