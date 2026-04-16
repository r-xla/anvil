# Primitive Print

Prints an array value to the console during execution and returns the
input unchanged. This is useful for debugging JIT-compiled code.

## Usage

``` r
nvl_print(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of any data type.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Returns `operand` as-is.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`stablehlo::hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html).

## See also

[`nv_print()`](https://r-xla.github.io/anvil/dev/reference/nv_print.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(1, 2, 3))
  nvl_print(x)
})
}
```
