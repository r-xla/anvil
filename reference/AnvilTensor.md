# AnvilTensor

Tensor objects in anvil that hold array data with automatic
differentiation support. Create tensors using `nv_tensor()`,
`nv_scalar()`, or `nv_empty()`.

## Usage

``` r
nv_tensor(data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL)

nv_scalar(data, dtype = NULL, device = NULL, ambiguous = NULL)

nv_empty(dtype, shape, device = NULL, ambiguous = FALSE)
```

## Arguments

- data:

  (any)  
  Object convertible to a
  [`PJRTBuffer`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html).

- dtype:

  (`NULL` \| `character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  One of pred, i8, i16, i32, i64, ui8, ui16, ui32, ui64, f32, f64 or a
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html).
  The default (`NULL`) uses `f32` for numeric data, `i32` for integer
  data, and `pred` for logical data.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The platform name for the tensor (`"cpu"`, `"cuda"`, `"metal"`).
  Default is to use the CPU, unless the data is already a
  [`PJRTBuffer`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html).
  You can change the default by setting the `PJRT_PLATFORM` environment
  variable.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape. The default (`NULL`) is to infer it from the data if possible.
  Note that `nv_tensor` interprets length 1 vectors as having shape
  `(1)`. To create a "scalar" with dimension `()`, use `nv_scalar`.

- ambiguous:

  (`NULL` \| `logical(1)`)  
  Whether the dtype should be marked as ambiguous. For `nv_tensor()`,
  defaults to `FALSE` (non-ambiguous) for new tensors, or preserves the
  existing value when `data` is already an `AnvilTensor`. For
  `nv_scalar()`, defaults to `TRUE` when `dtype` is `NULL` and data is
  numeric, `FALSE` otherwise.

## Value

(`AnvilTensor`) A tensor object.
