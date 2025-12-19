# Tensor-like Objects

A value that is either an [`AnvilTensor`](nv_tensor.md), can be
converted to it, or represents an abstract version of it. This also
includes atomic R vectors.

## See also

[nv_tensor](nv_tensor.md), [ConcreteTensor](ConcreteTensor.md),
[AbstractTensor](AbstractTensor.md), [LiteralTensor](LiteralTensor.md),
`GraphBox`

## Examples

``` r
if (FALSE) { # pjrt::plugin_is_downloaded()
x <- nv_tensor(1:4, dtype = "f32")
x
}
```
