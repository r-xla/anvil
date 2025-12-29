# Sample from a Discrete Uniform Distribution

Sample from a discrete distribution, analogous to R's
[`sample()`](https://rdrr.io/r/base/sample.html) function. Samples
integers from 1 to n with uniform probability and with replacement.

## Usage

``` r
nv_rdunif(shape, initial_state, n, dtype = "i32")
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  (\[`ui64[2]`\]\[tensorish\])  
  RNG state.

- n:

  (`integer(1)`)  
  Number of categories to sample from (samples integers 1 to n).

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the sampled integers.
