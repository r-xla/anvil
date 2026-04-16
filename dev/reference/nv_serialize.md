# Serialize arrays to raw bytes

Serializes a named list of arrays into the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

## Usage

``` r
nv_serialize(arrays, con = NULL)
```

## Arguments

- arrays:

  (named `list` of
  [`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md))  
  Named list of arrays to serialize. Names must be unique.

- con:

  (`NULL` \| connection)  
  An optional connection to write to. If `NULL` (default), a raw vector
  is returned.

## Value

A [`raw`](https://rdrr.io/r/base/raw.html) vector if `con` is `NULL`,
otherwise `NULL` (invisibly).

## Details

The ambiguity of the arrays is stored in the metadata and preserved in
write-read roundtrips.

## See also

[`nv_unserialize()`](https://r-xla.github.io/anvil/dev/reference/nv_unserialize.md),
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
