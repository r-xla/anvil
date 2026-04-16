# Deserialize arrays from raw bytes

Deserializes arrays from the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

## Usage

``` r
nv_unserialize(con, device = NULL, backend = default_backend())
```

## Arguments

- con:

  (connection \| [`raw`](https://rdrr.io/r/base/raw.html))  
  A connection or raw vector to read from.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device on which to place the loaded arrays (`"cpu"`, `"cuda"`,
  ...). Default is to use the CPU.

- backend:

  (`character(1)`)  
  Backend for the loaded arrays. Defaults to
  [`default_backend()`](https://r-xla.github.io/anvil/dev/reference/default_backend.md).

## Value

Named `list` of
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
objects.

## Details

The data type, shape, and
[ambiguity](https://r-xla.github.io/anvil/dev/reference/ambiguous.md) of
each array are restored from the serialized data.

## See also

[`nv_serialize()`](https://r-xla.github.io/anvil/dev/reference/nv_serialize.md),
[`nv_save()`](https://r-xla.github.io/anvil/dev/reference/nv_save.md),
[`nv_read()`](https://r-xla.github.io/anvil/dev/reference/nv_read.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded("cpu")
x <- nv_array(array(1:6, dim = c(2, 3)))
x
raw_data <- nv_serialize(list(x = x))
raw_data
nv_unserialize(raw_data)
}
```
