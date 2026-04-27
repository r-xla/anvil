# Convert to AnvlArray

Use this to canonicalize inputs at the start of a function so it works
both with eager executing and in combination with
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md). Use
`as_anvl_array()` for a single input and `as_anvl_arrays()` for multiple
inputs. The latter will also ensure all arrays are from the same backend
and live on the same device.

## Usage

``` r
as_anvl_array(x, device = NULL)

as_anvl_arrays(...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input to standardize.

- device:

  (`NULL` \|
  [`device`](https://r-xla.github.io/anvl/reference/device.md))  
  Target device. If `x` is an `AnvlArray` on a different device, an
  error is raised.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Inputs to align.

## Value

([`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md) for
`as_anvl_array()`, `list` of
[`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)s for
`as_anvl_arrays()`).

## Details

During tracing,
[boxes](https://r-xla.github.io/anvl/reference/GraphBox.md) are returned
as is. During eager mode, R literals and arrays are converted to
`AnvlArray`s on the specified device. For `AnvlArray` inputs, we check
that they live on provided device (if specified).

## Examples

``` r
as_anvl_array(1L)
#> AnvlArray
#>  1
#> [ CPUi32?{} ] 
as_anvl_arrays(nv_array(1:3), 1L)
#> [[1]]
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1
#> [ CPUi32?{} ] 
#> 
```
