# Array-like Objects

A `arrayish` value is any object that can be passed as an input to anvil
primitive functions such as
[`nvl_add`](https://r-xla.github.io/anvil/dev/reference/nvl_add.md) or
is an output of such a function.

During runtime, these are
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
objects.

The following types are arrayish (during compile-time):

- [`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md):
  a concrete array holding data on a device.

- [`GraphBox`](https://r-xla.github.io/anvil/dev/reference/GraphBox.md):
  a boxed abstract array representing a value in a graph.

- Literals: `numeric(1)`, `integer(1)`, `logical(1)`: promoted to scalar
  arrays.

Use `is_arrayish()` to check whether a value is arrayish.

## Usage

``` r
is_arrayish(x, literal = TRUE)
```

## Arguments

- x:

  (`any`)  
  Object to check.

- literal:

  (`logical(1)`)  
  Whether to accept R literals as arrayish.

## Value

`logical(1)`

## See also

[AnvilArray](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md),
[GraphBox](https://r-xla.github.io/anvil/dev/reference/GraphBox.md)

## Examples

``` r
# AnvilArrays are arrayish
is_arrayish(nv_array(1:4))
#> [1] TRUE

# Scalar R literals are arrayish by default
is_arrayish(1.5)
#> [1] TRUE

# Non-scalar vectors are not arrayish
is_arrayish(1:4)
#> [1] FALSE

is_arrayish(DebugBox(nv_aten("f32", c(2L, 3L))))
#> [1] TRUE

# Disable literal promotion
is_arrayish(1.5, literal = FALSE)
#> [1] FALSE
```
