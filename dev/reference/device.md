# Get the device of an array

Returns the device on which an array is allocated.

## Usage

``` r
device(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

[`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html)

## Details

This is implemented via the generic
[`tengen::device()`](https://r-xla.github.io/tengen/reference/device.html).

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
device(x)
#> <CpuDevice(id=0)>
```
