# AnvilArray

The main array object. Its type is determined by a data type and a
shape.

To transform arrays, apply
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)ted
functions. Directly calling operations (e.g. `nv_add(x, y)`) on
`AnvilArray` objects only performs type inference and returns an
[`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md)
– see
[`vignette("debugging")`](https://r-xla.github.io/anvil/dev/articles/debugging.md)
for details.

To compare whether two abstract arrays are equal, use
[`eq_type()`](https://r-xla.github.io/anvil/dev/reference/eq_type.md).

## Usage

``` r
nv_array(data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL)

nv_scalar(data, dtype = NULL, device = NULL, ambiguous = NULL)

nv_empty(dtype, shape, device = NULL, ambiguous = FALSE)
```

## Arguments

- data:

  (any)  
  Object convertible to a
  [`PJRTBuffer`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html).
  Includes [`integer()`](https://rdrr.io/r/base/integer.html),
  [`double()`](https://rdrr.io/r/base/double.html),
  [`logical()`](https://rdrr.io/r/base/logical.html) vectors and arrays.

- dtype:

  (`NULL` \| `character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  One of bool, i8, i16, i32, i64, ui8, ui16, ui32, ui64, f32, f64 or a
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html).
  The default (`NULL`) uses the current backend's default dtype: `f32`
  for numeric data on `"xla"`, `f64` for numeric data on `"quickr"`,
  `i32` for integer data, and `bool` for logical data.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device for the array (`"cpu"`, `"cuda"`). Default is to use the
  CPU for new arrays. This can be changed by setting the `PJRT_PLATFORM`
  environment variable.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The output shape of the array. The default (`NULL`) is to infer it
  from the data if possible. Note that `nv_array` interprets length 1
  vectors as having shape `(1)`. To create a "scalar" with dimension
  `()`, use `nv_scalar` or explicitly specify `shape = c()`.

- ambiguous:

  (`NULL` \| `logical(1)`)  
  Whether the dtype should be marked as ambiguous. Defaults to `FALSE`
  for new arrays.

## Value

(`AnvilArray`)

## Extractors

The following generic functions can be used to extract information from
an `AnvilArray`:

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html): Get
  the data type of the array.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html): Get
  the shape (dimensions) of the array.

- [`ndims()`](https://r-xla.github.io/tengen/reference/ndims.html): Get
  the number of dimensions.

- [`device()`](https://r-xla.github.io/tengen/reference/device.html):
  Get the device of the array.

- [`platform()`](https://r-xla.github.io/pjrt/reference/platform.html):
  Get the platform (e.g. `"cpu"`, `"cuda"`).

- [`ambiguous()`](https://r-xla.github.io/anvil/dev/reference/ambiguous.md):
  Get whether the dtype is ambiguous.

## Serialization

Arrays can be serialized to and from the
[safetensors](https://huggingface.co/docs/safetensors/index) format:

- [`nv_save()`](https://r-xla.github.io/anvil/dev/reference/nv_save.md)
  /
  [`nv_read()`](https://r-xla.github.io/anvil/dev/reference/nv_read.md):
  Save/load arrays to/from a file.

- [`nv_serialize()`](https://r-xla.github.io/anvil/dev/reference/nv_serialize.md)
  /
  [`nv_unserialize()`](https://r-xla.github.io/anvil/dev/reference/nv_unserialize.md):
  Serialize/deserialize arrays to/from raw vectors.

## See also

[nv_fill](https://r-xla.github.io/anvil/dev/reference/nv_fill.md),
[nv_iota](https://r-xla.github.io/anvil/dev/reference/nv_iota.md),
[nv_seq](https://r-xla.github.io/anvil/dev/reference/nv_seq.md),
[as_array](https://r-xla.github.io/anvil/dev/reference/as_array.md),
[nv_serialize](https://r-xla.github.io/anvil/dev/reference/nv_serialize.md)

## Examples

``` r
# A 1-d array (vector) with shape (4). Default type for integers is `i32`
nv_array(1:4)
#> AnvilArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUi32{4} ] 

# Specify a dtype
nv_array(c(1.5, 2.5, 3.5), dtype = "f64")
#> AnvilArray
#>  1.5000
#>  2.5000
#>  3.5000
#> [ CPUf64{3} ] 

# A 2x3 matrix
nv_array(1:6, shape = c(2L, 3L))
#> AnvilArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 

# A scalar array.
nv_scalar(3.14)
#> AnvilArray
#>  3.1400
#> [ CPUf32{} ] 

# A 0x3 array
nv_empty("f32", shape = c(0L, 3L))
#> AnvilArray
#> [ CPUf32{0,3} ] 

# --- Extractors ---
x <- nv_array(1:6, shape = c(2L, 3L))
dtype(x)
#> <i32>
shape(x)
#> [1] 2 3
ndims(x)
#> [1] 2
device(x)
#> <CpuDevice(id=0)>
platform(x)
#> [1] "cpu"
ambiguous(x)
#> [1] FALSE

# --- Transforming arrays with jit ---
add_one <- jit(function(x) x + 1)
add_one(nv_array(1:4))
#> AnvilArray
#>  2
#>  3
#>  4
#>  5
#> [ CPUf32?{4} ] 

# --- Debug mode (calling operations directly) ---
# Outside of jit, operations only perform type inference:
nv_add(nv_array(1:3), nv_array(4:6))
#> i32{3}
```
