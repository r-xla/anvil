# Internal: Random Unit Uniform Numbers

generate random uniform numbers in \[0, 1)

Sample from a uniform distribution in the open interval (lower, upper).

## Usage

``` r
nv_unif_rand(shape, initial_state, dtype = "f64")

nv_runif(shape, initial_state, dtype = "f32", lower = 0, upper = 1)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  (\[`ui64[2]`\]\[tensorish\])  
  RNG state.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- lower, upper:

  (`numeric(1)`)  
  Lower and upper bound.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the generated random numbers.
