# Random Discrete Sample

Sample from a discrete distribution, analogous to R's
[`sample()`](https://rdrr.io/r/base/sample.html) function. Samples
integers from 1 to n with uniform probability and with replacement.

## Usage

``` r
nv_sample_int(n, shape, initial_state, dtype = "i32")
```

## Arguments

- n:

  (`integer(1)`)  
  Number of categories to sample from (samples integers 1 to n).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`tensorish`](tensorish.md))  
  Tensor of type `ui64[2]` for the RNG state.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Output dtype (default "i32").

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the sampled integers (1 to
n).
