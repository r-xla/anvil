# Adding a Primitive

This guide explains how to implement a new primitive. It will primarily
focus on *how* to do this. See the [internals
vignette](https://r-xla.github.io/anvil/articles/internals.md) for more
information on how primitives work.

In general, there are two main reasons to add a new primitive:

1.  The operation is not expressible using existing primitives.
2.  The operation (or, for example, its derivative) would benefit from a
    dedicated implementation.

In order to be able to add a new primitive, it needs to be expressible
in the [{stablehlo}](https://github.com/r-xla/stablehlo) package. There
are various sceanrios:

1.  The operation is expressible in the StableHLO language and:
    1.  All required operations are already implemented in the
        {stablehlo} R package.

        \\\rightarrow\\ This is the simplest caseand the one we assume
        in this guide.

    2.  One or more required operations are not already implemented in
        the {stablehlo} R package.

        \\\rightarrow\\ You first need to implement the missing
        operations in the {stablehlo} R package. See [this
        issue](https://github.com/r-xla/stablehlo/issues/6) for which
        operations are missing and the [StableHLO
        specification](https://openxla.org/stablehlo/spec) for which are
        available.
2.  The operation is not available/cannot be efficiently expressed in
    StableHLO, because:
    1.  It requires shape dynamism (output shape depends on data, such
        as using boolean values for subsetting):

        \\\rightarrow\\ This is currently not possible, but we hope we
        can add support for this in the future, e.g. via a second,
        dynamic Fortran backend.

    2.  It cannot be expressed or can only expressed inefficiently:

        \\\rightarrow\\ You can implement a stableHLO custom call, see
        the custom print operation in
        [pjrt](https://github.com/r-xla/pjrt). This is currently not
        well documented, so you need to dig into the source code.

## Adding a Primitive: Practical Example

Let’s add a new primitive step by step. We’ll implement
`nvl_repeat_along` – a primitive that repeats a tensor multiple times
along a specified dimension.

For example, repeating `c(1, 2, 3)` twice along dimension 1 gives
`c(1, 2, 3, 1, 2, 3)`.

This primitive has a *dynamic* input (a tensor) and two *static*
parameters (how many times to repeat and which dimension).

### Step 1: Create the AnvilPrimitive Object

``` r
library(anvil)
p_repeat_along <- AnvilPrimitive("repeat_along")
```

This is simply the primitive object that will identify this operation in
an `AnvilGraph` and it will hold the rules for how to lower it to
StableHLO and how to compute gradients.

### Step 2: Define the nvl\_\* Function

The `nvl_*` function does two things:

1.  Defines an **inference function** that computes output types from
    input types
2.  Calls
    [`graph_desc_add()`](https://r-xla.github.io/anvil/reference/graph_desc_add.md)
    to record the operation in the current `GraphDescriptor`.

``` r
nvl_repeat_along <- function(operand, times, dim) {
  # type of operand is checked by graph_desc_add()
  infer_fn <- function(operand, times, dim) {
    if (!checkmate::test_integerish(dim, lower = 1, upper = ndims(operand), len = 1L)) {
      cli::cli_abort("{.arg dim} must be between 1 and {ndims(operand)}, but is {.val dim}")
    }
    if (!checkmate::test_integerish(times, lower = 1, len = 1L)) {
      cli_abort("times must be a positive integer, but is {times}")
    }
    new_shape <- shape(operand)
    new_shape[dim] <- new_shape[dim] * times
    list(AbstractTensor(
      dtype = dtype(operand),
      shape = Shape(new_shape),
      ambiguous = operand$ambiguous
    ))
  }

  graph_desc_add(
    p_repeat_along,             # The primitive
    list(operand = operand),    # Dynamic inputs (tensors)
    params = list(              # Static parameters
      times = times,
      dim = dim
    ),
    infer_fn = infer_fn
  )[[1L]]  # Extract single output from list
}
```

Key points:

- The inference function receives abstract tensors (types, not values)
  for dynamic inputs, and actual values for static parameters. It must
  verify that the arguments to the function are valid. Also, it should
  throw clear error messages if the arguments are invalid, as {anvil}
  programs are otherwise hard to debug.
- [`graph_desc_add()`](https://r-xla.github.io/anvil/reference/graph_desc_add.md)
  returns a list of outputs; use `[[1L]]` for single-output primitives.
- Propagate the `ambiguous` flag from inputs to outputs, see [type
  promotion](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for what this means.

If the primitive is a wrapper around a stablehlo operation, it is
possible to use the corresponding inference function from the stablehlo
package (such as
[`stablehlo::infer_types_concatenate`](https://r-xla.github.io/stablehlo/reference/hlo_concatenate.html)).
When doing so, you need to:

1.  Convert the abstract tensors to stablehlo `ValueType`s using
    [`at2vt()`](https://r-xla.github.io/anvil/reference/at2vt.md).
2.  Call the stablehlo inference function and obtain a list of
    `ValueType`s.
3.  Convert the `ValueType`s back to abstract tensors using
    [`vt2at()`](https://r-xla.github.io/anvil/reference/vt2at.md).
4.  Set the `ambiguous` flag of the output depending on the inputs
    (`ambiguity` is strictly an {anvil} concept, not a stablehlo
    concept).

### Step 3: Add the StableHLO Rule

The StableHLO rule defines how to lower this primitive to actual
operations, which is used in the
[`stablehlo()`](https://r-xla.github.io/anvil/reference/stablehlo.md)
lowering pass. We implement `repeat_along` using concatenation:

``` r
p_repeat_along[["stablehlo"]] <- function(operand, times, dim) {
  operands <- rep(list(operand), times)
  list(rlang::exec(stablehlo::hlo_concatenate, !!!operands, dimension = dim - 1L))
}
```

The rule receives:

- Dynamic inputs as
  [`stablehlo::FuncValue`](https://r-xla.github.io/stablehlo/reference/FuncValue.html)s.
- Static parameters as their R values.

It must return a list of
[`stablehlo::FuncValue`](https://r-xla.github.io/stablehlo/reference/FuncValue.html)s,
even if there is only one output.

**Important**: StableHLO uses 0-based indexing, while {anvil} uses R’s
1-based indexing. Always convert dimension indices by subtracting 1.
Also note that in
[`graph_desc_add()`](https://r-xla.github.io/anvil/reference/graph_desc_add.md)
we are converting the error messages from stablehlo to our 1-based
indexing, so you do not have to worry about that here.

### Step 4: Add the Backward Rule

If the operation should support automatic differentiation, add a
backward rule. The idea here is the following, where we assume the input
`operand` has shape `(s_1, ..., s_n)`, which means that the output (and
therefore it’s gradient) has shape
`(s_1, ..., s_{dim-1}, s_dim * times, s_{dim+1}, ..., s_n)`.

1.  Reshape the gradient to
    `(s_1, ..., s_{dim-1}, s_dim, times, s_{dim+1}, ..., s_n)`.
2.  Sum over the `times` dimension and drop the `times` dimension.

``` r
p_repeat_along[["backward"]] <- function(inputs, outputs, grads, dim, times, .required) {
  if (!.required[[1L]]) {
    return(list(NULL))
  }

  grad <- grads[[1L]]
  operand <- inputs[[1L]]

  old_shape <- shape(operand)
  grad_shape <- shape(grad)

  new_shape <- grad_shape
  new_shape[dim] <- old_shape[dim]
  new_shape <- append(new_shape, times, after = dim - 1L)

  grad_reshaped <- nvl_reshape(grad, new_shape)
  grad_summed <- nvl_reduce_sum(grad_reshaped, dims = dim, drop = TRUE)
  list(grad_summed)
}
```

The backward rule receives:

- `inputs`: Input `GraphValue`s from the forward pass
- `outputs`: Output `GraphValue`s from the forward pass
- `grads`: Gradients flowing back from downstream (one per output)
- Static parameters by name (here: `dim`, `times`)
- `.required`: Logical vector indicating which input gradients are
  needed

It returns a list with one gradient per input (or `NULL` if not
required).

## Step 5: Register the Primitive

To not pollute the global namespace, the primitive objects
(`p_repeat_along` in our case) are not exported. Instead, they can be
retrieved via `prim("repeat_along")`. To make this work, you need to
register the primitive object:

``` r
register_primitive("repeat_along", p_repeat_along)
prim("repeat_along")
#> <AnvilPrimitive:repeat_along>
```

### Step 6: Add an `nv_` API Function

In {anvil}, we also offer convenience wrappers around the primitives. An
example is `nvl_add` vs `nv_add`, where the latter calls into the former
after optionally broadcasting (scalar) inputs:

``` r
nv_add(1L, nv_tensor(2:3))
#> i32{2}
nvl_add(1L, nv_tensor(2:3))
#> Error in `nvl_add()`:
#> ! `lhs` and `rhs` must have the same tensor type.
#> ✖ Got tensor<i32> and tensor<2xi32>.
```

In our case, no such convenience is needed and the functionality is not
too low-level (for it to be generally useful), so we can just reassign
the `nvl_*` function to an `nv_*` function:

``` r
nv_repeat_along <- nvl_repeat_along
```

Note that in the `nv_*` wrapper function, you can only access certain
properties of the input tensorish values via:

- [`shape_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)
- [`ndims_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)
- [`dtype_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)
- [`ambiguous_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)

If you, for example, use
[`shape()`](https://r-xla.github.io/anvil/reference/shape.md) instead of
[`shape_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md),
your function won’t work with R literals. I.e., `<extract>_abstract()`
first converts the input to an `AbstractTensor` (if possible) and then
extracts the property.

### Using Your Primitive

You can now use the primitive:

``` r
x <- nv_tensor(c(1, 2, 3), shape = c(3, 1))
result <- jit(function(x) nvl_repeat_along(x, times = 2L, dim = 2L))(x)
result
#> AnvilTensor
#>  1 1
#>  2 2
#>  3 3
#> [ CPUf32{3,2} ]
```

And compute gradients through it.

``` r
f <- function(x) {
  repeated <- nvl_repeat_along(x, times = 2L, dim = 2L)
  sum(repeated)
}

grad_f <- jit(gradient(f))
grad_f(x)[[1L]]
#> AnvilTensor
#>  2
#>  2
#>  2
#> [ CPUf32{3,1} ]
```

## Contributing to the Package

If you want to contribute your primitive to {anvil}, there are some
additional things to be aware of.

### File Organization

- **`R/primitives.R`**: Define `AnvilPrimitive` object and `nvl_*`
  function
- **`R/rules-stablehlo.R`**: Add the StableHLO lowering rule
- **`R/rules-backward.R`**: Add the backward rule (if differentiable)
- **`R/api.R`**: Add the `nv_*` wrapper function (or possibly in another
  **api** file).

### Testing

Tests can go in two places:

1.  **`inst/extra-tests/`**: For tests that compare against torch. These
    live in `inst/` to avoid listing torch as a dependency.
2.  **`tests/testthat/`**: For tests without a torch counterpart.

**Important**: Test names must start with `"p_<name>"` (matching the
primitive object name). The meta tests verify that every primitive has
corresponding tests.

Since no torch counterpart exists for `nvl_repeat_along`, we would add
manual tests in:

- `tests/testthat/test-primitives-stablehlo.R`
- `tests/testthat/test-primitives-backward.R`

Also, ensure that no linter errors are present, `devtools::check()`
passes, and format the code using `make format`.

## Higher-Order Primitives

Higher-Order Primitives are primitives that parameterized by an R
function or expression. Examples include `nvl_if` and `nvl_while`. These
are generally much more complex to handle, so we don’t cover them here
in detail (for now). The general idea, however, is that the primitive
`nvl_*` function needs to trace the provided function using
[`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md) and
then forward this graph to the stablehlo lowering rule and backward
rule.
