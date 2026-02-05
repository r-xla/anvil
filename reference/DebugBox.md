# Debug Box Class

[`AnvilBox`](https://r-xla.github.io/anvil/reference/AnvilBox.md)
subclass that wraps an
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
for use in debug mode. When anvil operations (e.g.
[`nv_add()`](https://r-xla.github.io/anvil/reference/nv_add.md)) are
called outside of
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md), they return
`DebugBox` objects instead of actual computed results. This allows
checking the types and shapes of intermediate values without compiling
or running a computation – see
[`vignette("debugging")`](https://r-xla.github.io/anvil/articles/debugging.md)
for details.

The convenience constructor
[`debug_box()`](https://r-xla.github.io/anvil/reference/debug_box.md)
creates a `DebugBox` from a dtype and shape directly.

## Usage

``` r
DebugBox(aval)
```

## Arguments

- aval:

  ([`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md))  
  The abstract tensor representing the value.

## Value

(`DebugBox`)

## Extractors

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html)

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html)

- [`ndims()`](https://r-xla.github.io/tengen/reference/ndims.html)

- [`ambiguous()`](https://r-xla.github.io/anvil/reference/ambiguous.md)

## See also

[AnvilBox](https://r-xla.github.io/anvil/reference/AnvilBox.md),
[GraphBox](https://r-xla.github.io/anvil/reference/GraphBox.md),
[`debug_box()`](https://r-xla.github.io/anvil/reference/debug_box.md),
[AbstractTensor](https://r-xla.github.io/anvil/reference/AbstractTensor.md)

## Examples

``` r
x <- nv_tensor(1:4)
y <- nv_tensor(5:8)
result <- nv_add(x, y)
result
#> i32{4}
dtype(result)
#> <i32>
shape(result)
#> [1] 4

# Create directly via debug_box()
db <- debug_box("f32", c(2L, 3L))
db
#> f32{2,3}
nv_reduce_sum(db, dims = 2L)
#> f32{2}
```
