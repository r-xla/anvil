# AnvilTensor

The main tensor object. Its type is determined by a data type and a
shape.

To transform tensors, apply
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md)ted functions.
Directly calling operations (e.g. `nv_add(x, y)`) on `AnvilTensor`
objects only performs type inference and returns an
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
– see
[`vignette("debugging")`](https://r-xla.github.io/anvil/articles/debugging.md)
for details.

To compare whether two abstract tensors are equal, use
[`eq_type()`](https://r-xla.github.io/anvil/reference/eq_type.md).

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
  Includes [`integer()`](https://rdrr.io/r/base/integer.html),
  [`double()`](https://rdrr.io/r/base/double.html),
  [`logical()`](https://rdrr.io/r/base/logical.html) vectors and arrays.

- dtype:

  (`NULL` \| `character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  One of pred, i8, i16, i32, i64, ui8, ui16, ui32, ui64, f32, f64 or a
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html).
  The default (`NULL`) uses `f32` for numeric data, `i32` for integer
  data, and `i1` for logical data.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device for the tensor (`"cpu"`, `"cuda"`). Default is to use the
  CPU for new tensors. This can be changed by setting the
  `PJRT_PLATFORM` environment variable.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The output shape of the tensor. The default (`NULL`) is to infer it
  from the data if possible. Note that `nv_tensor` interprets length 1
  vectors as having shape `(1)`. To create a "scalar" with dimension
  `()`, use `nv_scalar` or explicitly specify `shape = c()`.

- ambiguous:

  (`NULL` \| `logical(1)`)  
  Whether the dtype should be marked as ambiguous. Defaults to `FALSE`
  for new tensors.

## Value

(`AnvilTensor`)

## Extractors

The following generic functions can be used to extract information from
an `AnvilTensor`:

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html): Get
  the data type of the tensor.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html): Get
  the shape (dimensions) of the tensor.

- [`ndims()`](https://r-xla.github.io/tengen/reference/ndims.html): Get
  the number of dimensions.

- [`device()`](https://r-xla.github.io/tengen/reference/device.html):
  Get the device of the tensor.

- [`platform()`](https://r-xla.github.io/pjrt/reference/platform.html):
  Get the platform (e.g. `"cpu"`, `"cuda"`).

- [`ambiguous()`](https://r-xla.github.io/anvil/reference/ambiguous.md):
  Get whether the dtype is ambiguous.

## Serialization

Tensors can be serialized to and from the
[safetensors](https://huggingface.co/docs/safetensors/index) format:

- [`nv_write()`](https://r-xla.github.io/anvil/reference/nv_serialization.md)
  /
  [`nv_read()`](https://r-xla.github.io/anvil/reference/nv_serialization.md):
  Save/load tensors to/from a file.

- [`nv_serialize()`](https://r-xla.github.io/anvil/reference/nv_serialization.md)
  /
  [`nv_unserialize()`](https://r-xla.github.io/anvil/reference/nv_serialization.md):
  Serialize/deserialize tensors to/from raw vectors.

## See also

nv_fill, nv_iota, nv_seq, as_array,
[nv_serialization](https://r-xla.github.io/anvil/reference/nv_serialization.md)

## Examples

``` r
# A 1-d tensor (vector) with shape (4). Default type for integers is `i32`
nv_tensor(1:4)
#> AnvilTensor
#>  1
#>  2
#>  3
#>  4
#> [ CPUi32{4} ] 

# Specify a dtype
nv_tensor(c(1.5, 2.5, 3.5), dtype = "f64")
#> AnvilTensor
#>  1.5000
#>  2.5000
#>  3.5000
#> [ CPUf64{3} ] 

# A 2x3 matrix
nv_tensor(1:6, shape = c(2L, 3L))
#> AnvilTensor
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 

# A scalar tensor.
nv_scalar(3.14)
#> AnvilTensor
#>  3.1400
#> [ CPUf32{} ] 

# A 0x3 tensor
nv_empty("f32", shape = c(0L, 3L))
#> AnvilTensor
#> [ CPUf32{0,3} ] 

# --- Extractors ---
x <- nv_tensor(1:6, shape = c(2L, 3L))
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

# --- Transforming tensors with jit ---
add_one <- jit(function(x) x + 1)
add_one(nv_tensor(1:4))
#> AnvilTensor
#>  2
#>  3
#>  4
#>  5
#> [ CPUf32?{4} ] 

# --- Debug mode (calling operations directly) ---
# Outside of jit, operations only perform type inference:
nv_add(nv_tensor(1:3), nv_tensor(4:6))
#> i32{3}
```
