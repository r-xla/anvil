# Primitive RNG Bit Generator

Generates random bits using the specified algorithm.

## Usage

``` r
nvl_rng_bit_generator(initial_state, rng_algorithm = "THREE_FRY", dtype, shape)
```

## Arguments

- initial_state:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  RNG state tensor.

- rng_algorithm:

  (`character(1)`)  
  Algorithm name (default "THREE_FRY").

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Output shape.

## Value

List of new state and random tensor.
