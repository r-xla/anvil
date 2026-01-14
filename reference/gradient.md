# Gradient

Transform a function to its gradient.

## Usage

``` r
gradient(f, wrt = NULL, static_args)

value_and_gradient(f, wrt = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to compute the gradient of.

- wrt:

  (`character`)  
  Names of the arguments to compute the gradient with respect to.

- static_args:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Values for the static (non-tensor) arguments.

## Functions

- `value_and_gradient()`: Returns both the value and the gradient
