# Internal: Random Unit Uniform Numbers

generate random uniform numbers in \[0, 1)

generate random uniform numbers in \]lower, upper\[

## Usage

``` r
nv_unif_rand(initial_state, dtype = "f64", shape)

nv_runif(initial_state, dtype = "f32", shape, lower = 0, upper = 1)
```

## Arguments

- initial_state:

  ([`tensorish`](tensorish.md))  
  Tensor of type `ui64[2]`.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

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
