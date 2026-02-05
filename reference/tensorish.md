# Tensor-like Objects

A `tensorish` value is any object that can be passed as an input to
anvil primitive functions such as
[`nvl_add`](https://r-xla.github.io/anvil/reference/nvl_add.md) or is an
output of such a function.

During runtime, these are
[`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
objects.

The following types are tensorish (during compile-time):

- [`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md):
  a concrete tensor holding data on a device.

- [`GraphBox`](https://r-xla.github.io/anvil/reference/GraphBox.md): a
  boxed abstract tensor representing a value in a graph.

- Literals: `numeric(1)`, `integer(1)`, `logical(1)`: promoted to scalar
  tensors.

Use `is_tensorish()` to check whether a value is tensorish.

## Usage

``` r
is_tensorish(x, literal = TRUE)
```

## Arguments

- x:

  (`any`)  
  Object to check.

- literal:

  (`logical(1)`)  
  Whether to accept R literals as tensorish.

## Value

`logical(1)`

## See also

[AnvilTensor](https://r-xla.github.io/anvil/reference/AnvilTensor.md),
[GraphBox](https://r-xla.github.io/anvil/reference/GraphBox.md)

## Examples

``` r
# AnvilTensors are tensorish
is_tensorish(nv_tensor(1:4))
#> [1] TRUE

# Scalar R literals are tensorish by default
is_tensorish(1.5)
#> [1] TRUE

# Non-scalar vectors are not tensorish
is_tensorish(1:4)
#> [1] FALSE

is_tensorish(DebugBox(nv_aten("f32", c(2L, 3L))))
#> [1] TRUE

# Disable literal promotion
is_tensorish(1.5, literal = FALSE)
#> [1] FALSE
```
