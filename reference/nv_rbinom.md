# Sample from a Binomial Distribution

Sample from a binomial distribution with \$n\$ trials and success
probability \$p\$.

## Usage

``` r
nv_rbinom(shape, initial_state, n = 1L, prob = 0.5, dtype = "i32")
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
  Number of trials. Default is 1 (Bernoulli).

- prob:

  (`numeric(1)`)  
  Probability of success on each trial. Default is 0.5.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the generated random samples.
