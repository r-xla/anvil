# Random Uniform Numbers

generate random uniform numbers in \]lower, upper\[

generate random uniform numbers in \[0, 1)

## Usage

``` r
nv_runif(initial_state, dtype = "f64", shape, lower = 0, upper = 1)

nv_unif_rand(initial_state, dtype = "f64", shape)
```

## Arguments

- initial_state:

  state seed

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Output dtype either "f32" or "f64"

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- lower, upper:

  (`numeric(1)`)  
  Lower and upper bound.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the generated random numbers.
