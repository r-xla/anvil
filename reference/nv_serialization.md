# Tensor serialization and I/O

Read and write tensors using the safetensors format. to/from raw vectors
in memory.

## Usage

``` r
nv_write(tensors, path)

nv_read(path, device = NULL)

nv_serialize(tensors)

nv_unserialize(con, device = NULL)
```

## Arguments

- tensors:

  (named `list` of [`AnvilTensor`](AnvilTensor.md))  
  Named list of tensors.

- path:

  (`character(1)`)  
  Path to the safetensors file.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The platform name for the tensor (`"cpu"`, `"cuda"`, ...). Default is
  to use the CPU.

- con:

  (connection)  
  A connection object to read serialized tensors from.

## Value

- `nv_write()`: `NULL` (invisibly)

- `nv_read()`: Named list of [`AnvilTensor`](AnvilTensor.md) objects

- `nv_serialize()`: Raw vector containing serialized tensors

- `nv_unserialize()`: Named list of [`AnvilTensor`](AnvilTensor.md)
  objects

## Details

These functions wrap the safetensors format functionality provided by
the [`safetensors`](https://github.com/mlverse/safetensors) package.

## Examples

``` r
x <- nv_tensor(array(1:6, dim = c(2, 3)))
raw_data <- nv_serialize(list(x = x))
reloaded <- nv_unserialize(raw_data)
expect_equal(x, reloaded$x)
#> Error in expect_equal(x, reloaded$x): could not find function "expect_equal"
```
