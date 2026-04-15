# Quickr device

Device descriptor for the quickr backend. The only supported `type` is
`"cpu"`.

## Usage

``` r
quickr_device(x = "cpu")
```

## Arguments

- x:

  (`character(1)`)  
  Device type. Currently only supports `"cpu"`.

## Value

A `QuickrDevice` object.

## See also

[`nv_device()`](https://r-xla.github.io/anvil/dev/reference/nv_device.md),
[`AnvilBackendQuickr()`](https://r-xla.github.io/anvil/dev/reference/AnvilBackendQuickr.md).
