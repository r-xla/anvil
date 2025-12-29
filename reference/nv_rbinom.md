# Random Binomial Samples

Generate random Binomial(1, 0.5) samples (0 or 1) by extracting
individual bits from the random number generator. This is equivalent to
Bernoulli(0.5) samples.

## Usage

``` r
nv_rbinom(initial_state, dtype = "i32", shape)
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

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the generated random samples
(0 or 1).
