# Gradient

Transform a function to its gradient.

## Usage

``` r
gradient(f, wrt = NULL)

value_and_gradient(f, wrt = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to compute the gradient of.

- wrt:

  (`character` or `NULL`)  
  Names of the arguments to compute the gradient with respect to. If
  `NULL` (the default), the gradient is computed with respect to all
  arguments.

## Functions

- `value_and_gradient()`: Returns both the value and the gradient

## Ambiguity

When performing the backward pass, the ambiguity does not really play a
role anymore. This is because we don't call into `nv_`-functions anymore
that promote non-matching tensors, but only primitive `nvl_`-functions.
The promotion has already be done in the forward pass. We still want to
ensure, however, that the gradient values have the same ambiguity as the
input values. This is simply achieved by setting the ambiguity at the
end of the backward pass. The ambiguity of a single backward rule
therefore does not really matter.
