# Array-like Objects

A `arrayish` value is any object that can be input to a primitive such
as [`nvl_add`](https://r-xla.github.io/anvil/dev/reference/nvl_add.md).

During runtime of a JIT-compiled function, these are
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
objects.

The following types are arrayish (during tracing):

- [`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md):
  a concrete array holding data on a device.

- [`GraphBox`](https://r-xla.github.io/anvil/dev/reference/GraphBox.md):
  a boxed abstract array representing a value in a graph.

- Length-1 vectors: `numeric(1)` and `logical(1)`

- R arrays of types: `numeric` and `logical`.

Use `is_arrayish()` to check whether a value is arrayish.

## Usage

``` r
is_arrayish(x, convert_ok = TRUE)
```

## Arguments

- x:

  (`any`)  
  Object to check.

- convert_ok:

  (`logical(1)`)  
  Whether to accept `numeric(1)` and `logical(1)` and R arrays of type
  `numeric` and `logical`.

## Value

`logical(1)`

## See also

[AnvilArray](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md),
[GraphBox](https://r-xla.github.io/anvil/dev/reference/GraphBox.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
# AnvilArrays are arrayish
is_arrayish(nv_array(1:4))

# Scalar R literals are arrayish by default
is_arrayish(1.5)
# R arrays are arrayish by default
is_arrayish(array(1.5))

# R arrays
is_arrayish(array(1:4), convert_ok = TRUE)
is_arrayish(array(1:4), convert_ok = FALSE)

# Length 1 vectors
is_arrayish(1.5, convert_ok = FALSE)
is_arrayish(1.5, convert_ok = TRUE)
}
```
