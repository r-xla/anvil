# Ahead-of-time compile a function to XLA

Compiles a function to an XLA executable via tracing.

Returns a callable R function that executes the compiled binary. Unlike
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md), compilation
happens eagerly at definition time rather than on first call, so the
input shapes and dtypes must be specified upfront via abstract tensors
(see
[`nv_aten()`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)).

## Usage

``` r
xla(f, args, donate = character(), device = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to compile. Must accept and return
  [`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)s.

- args:

  (`list`)  
  List of abstract tensor specifications (e.g. from
  [`nv_aten()`](https://r-xla.github.io/anvil/reference/AbstractTensor.md))
  describing the expected shapes and dtypes of `f`'s arguments.

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated.

- device:

  (`character(1)`)  
  Target device such as `"cpu"` (default) or `"cuda"`.

## Value

(`function`)  
A function that accepts
[`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
arguments (matching the flat inputs) and returns the result as
[`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)s.

## Details

Traces `f` with the given abstract `args` (via
[`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md)),
lowers the resulting graph via
[`stablehlo()`](https://r-xla.github.io/anvil/reference/stablehlo.md)
and then compiles it to an XLA executable via
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).
and compiles it to an XLA executable immediately.

## See also

[`jit()`](https://r-xla.github.io/anvil/reference/jit.md) for lazy
compilation,
[`compile_to_xla()`](https://r-xla.github.io/anvil/reference/compile_to_xla.md)
for the lower-level API.

## Examples

``` r
f_compiled <- xla(function(x, y) x + y,
  args = list(x = nv_aten("f32", c(2, 2)), y = nv_aten("f32", c(2, 2)))
)
a <- nv_tensor(array(1:4, c(2, 2)), dtype = "f32")
b <- nv_tensor(array(5:8, c(2, 2)), dtype = "f32")
f_compiled(a, b)
#> AnvilTensor
#>   6 10
#>   8 12
#> [ CPUf32{2,2} ] 
```
