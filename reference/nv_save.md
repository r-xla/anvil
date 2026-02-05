# Save tensors to a file

Saves a named list of tensors to a file in the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

## Usage

``` r
nv_save(tensors, path)
```

## Arguments

- tensors:

  (named `list` of
  [`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md))  
  Named list of tensors to save. Names must be unique.

- path:

  (`character(1)`)  
  File path to write to.

## Value

`NULL` (invisibly).

## Details

This is a convenience wrapper around
[`nv_serialize()`](https://r-xla.github.io/anvil/reference/nv_serialize.md)
that opens and closes a file connection.

## See also

[`nv_read()`](https://r-xla.github.io/anvil/reference/nv_read.md),
[`nv_serialize()`](https://r-xla.github.io/anvil/reference/nv_serialize.md),
[`nv_unserialize()`](https://r-xla.github.io/anvil/reference/nv_unserialize.md)

## Examples

``` r
x <- nv_tensor(array(1:6, dim = c(2, 3)))
x
#> AnvilTensor
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
path <- tempfile(fileext = ".safetensors")
nv_save(list(x = x), path)
nv_read(path)
#> $x
#> AnvilTensor
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
#> 
```
