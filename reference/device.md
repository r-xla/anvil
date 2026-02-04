# Get the device of a tensor

Returns the device on which a tensor is allocated.

## Usage

``` r
device(x, ...)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  A tensor-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

[`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html)

## Details

This is implemented via the generic
[`tengen::device()`](https://r-xla.github.io/tengen/reference/device.html).

## Examples

``` r
x <- nv_tensor(1:4, dtype = "f32")
device(x)
#> <CpuDevice(id=0)>
```
