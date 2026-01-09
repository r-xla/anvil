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
raw_data
#>  [1] 39 00 00 00 00 00 00 00 7b 22 78 22 3a 7b 22 73 68 61 70 65 22 3a 5b 32 2c
#> [26] 33 5d 2c 22 64 74 79 70 65 22 3a 22 49 33 32 22 2c 22 64 61 74 61 5f 6f 66
#> [51] 66 73 65 74 73 22 3a 5b 30 2c 32 34 5d 7d 7d 01 00 00 00 03 00 00 00 05 00
#> [76] 00 00 02 00 00 00 04 00 00 00 06 00 00 00
reloaded <- nv_unserialize(raw_data)
reloaded
#> $x
#> AnvilTensor 
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2x3} ] 
#> 
```
