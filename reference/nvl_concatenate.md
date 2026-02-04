# Primitive Concatenate

Concatenates tensors along a dimension. For a more user-friendly
interface, see
[`nv_concatenate()`](https://r-xla.github.io/anvil/reference/nv_concatenate.md),
which also handles type promotion and scalar broadcasting.

## Usage

``` r
nvl_concatenate(..., dimension)
```

## Arguments

- ...:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensors to concatenate. Must all have the same data type and rank. All
  dimensions must match except along `dimension`.

- dimension:

  (`integer(1)`)  
  Dimension along which to concatenate (1-indexed).

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as the inputs. The output shape matches the
inputs in all dimensions except `dimension`, which is the sum of the
input sizes along that dimension. It is ambiguous if all inputs are
ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## Shapes

All inputs must have the same rank and shape except along `dimension`.
The output dimension size along `dimension` is the sum of the input
dimension sizes along `dimension`.

## StableHLO

Lowers to
[`stablehlo::hlo_concatenate()`](https://r-xla.github.io/stablehlo/reference/hlo_concatenate.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(4, 5, 6))
  nvl_concatenate(x, y, dimension = 1L)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUf32{6} ] 
```
