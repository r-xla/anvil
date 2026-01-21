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
