# Assert Shape Vector

Check whether an input is a valid shape vector (integer vector with all
positive values).

## Usage

``` r
assert_shapevec(x, min_len = 1L, var_name = rlang::caller_arg(x))
```

## Arguments

- x:

  Object to check.

- min_len:

  (`integer(1)`)  
  Minimum length of the shape vector. Default is 1.

- var_name:

  (`character(1)`)  
  Name of the variable to use in error messages.

## Value

Invisibly returns `x` if the assertion passes.
