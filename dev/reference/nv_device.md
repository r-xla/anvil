# Create a Device

Constructs a backend-specific device object.

A device identifies a compute resources, such as CPU, or a specific GPU.
It is relevant for data allocation (e.g. via
[`nv_array()`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md))
but also compilation
([jit](https://r-xla.github.io/anvil/dev/reference/jit.md)).

## Usage

``` r
nv_device(x, backend = default_backend())
```

## Arguments

- x:

  (`character(1)`)  
  Identifier for the device. E.g. `"cpu"`, `"cuda"`, or `"cuda:<n>"`
  (for the n-th GPU).

- backend:

  (`character(1)`)  
  The backend for which to create the device.

## Value

A backend-specific device object (e.g. `PJRTDevice` for `"xla"`,
[`quickr_device`](https://r-xla.github.io/anvil/dev/reference/quickr_device.md)
for `"quickr"`).

## See also

[`backend()`](https://r-xla.github.io/anvil/dev/reference/backend.md),
[`AnvilBackend()`](https://r-xla.github.io/anvil/dev/reference/AnvilBackend.md).

## Examples

``` r
# Create CPU device for xla backend:
nv_device("cpu", "xla")
#> <CpuDevice(id=0)>
# Create CPU device for quickr backend:
nv_device("cpu", "quickr")
#> QuickrDevice(cpu) 
```
