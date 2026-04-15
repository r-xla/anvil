# Create a backend

Create a backend

## Usage

``` r
AnvilBackend(
  data_constructor,
  dtype,
  shape,
  ambiguous,
  as_array,
  as_raw,
  platform,
  device,
  new_device,
  print_data,
  jit
)
```

## Arguments

- data_constructor:

  (`function`)  
  Constructs an AnvilArray from R data. This should be a
  [`structure()`](https://rdrr.io/r/base/structure.html) with at least a
  `$data` field that contains the actual underlying data (`PJRTBuffer`
  for `"xla"` backend, [`array()`](https://rdrr.io/r/base/array.html)
  for `"quickr"` backend).

- dtype:

  (`function`)  
  Extracts the dtype from an AnvilArray.

- shape:

  (`function`)  
  Extracts the shape from an AnvilArray.

- ambiguous:

  (`function`)  
  Extracts the ambiguous flag from an AnvilArray.

- as_array:

  (`function`)  
  Converts an AnvilArray to an R array.

- as_raw:

  (`function`)  
  Converts an AnvilArray to raw bytes.

- platform:

  (`function`)  
  Returns the platform name (e.g. `"cpu"`).

- device:

  (`function`)  
  Returns the device object for an AnvilArray.

- new_device:

  (`function`)  
  Constructs a backend-specific device object from a device type string
  (e.g. `"cpu"`). Called by
  [`nv_device()`](https://r-xla.github.io/anvil/dev/reference/nv_device.md).

- print_data:

  (`function`)  
  Prints the array data with a footer.

- jit:

  (`function`)  
  Creates a JIT-compiled function implementation.

## Value

An `AnvilBackend` object.
