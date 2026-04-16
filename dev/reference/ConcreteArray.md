# Concrete Array Class

An
[`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md)
that also holds a reference to the actual array data. Usually represents
a closed-over constant in a program. Inherits from
[`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md).

## Usage

``` r
ConcreteArray(data)
```

## Arguments

- data:

  ([`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md))  
  The actual array data.

## Lowering

When lowering to XLA, these become inputs to the executable instead of
embedding them into programs as constants. This is to avoid increasing
compilation time and bloating the size of the executable.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
y <- nv_array(c(0.5, 0.6))
x <- ConcreteArray(y)
x
ambiguous(x)
shape(x)
ndims(x)
dtype(x)

# How it appears during tracing
graph <- trace_fn(function() y, list())
graph
graph$outputs[[1]]$aval
}
```
