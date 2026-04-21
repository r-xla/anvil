# Convert to AnvilArray

Use this to canonicalize inputs at the start of a function so it works
both with eager executing and in combination with
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md). Use
`as_anvil_array()` for a single input and `as_anvil_arrays()` for
multiple inputs. The latter will also ensure all arrays are from the
same backend and live on the same device.

## Usage

``` r
as_anvil_array(x, device = NULL)

as_anvil_arrays(...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Input to standardize.

- device:

  (`NULL` \|
  [`device`](https://r-xla.github.io/anvil/dev/reference/device.md))  
  Target device. If `x` is an `AnvilArray` on a different device, an
  error is raised.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Inputs to align.

## Value

([`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
for `as_anvil_array()`, `list` of
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)s
for `as_anvil_arrays()`).

## Details

During tracing,
[boxes](https://r-xla.github.io/anvil/dev/reference/GraphBox.md) are
returned as is. During eager mode, R literals and arrays are converted
to `AnvilArray`s on the specified device. For `AnvilArray` inputs, we
check that they live on provided device (if specified).

## Examples

``` r
as_anvil_array(1L)
#> AnvilArray
#>  1
#> [ CPUi32?{} ] 
as_anvil_arrays(nv_array(1:3), 1L)
#> [[1]]
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
#> 
#> [[2]]
#> AnvilArray
#>  1
#> [ CPUi32?{} ] 
#> 
```
