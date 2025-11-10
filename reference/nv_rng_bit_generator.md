# Random Numbers

generate random bits of desired shape and dtype

## Usage

``` r
nv_rng_bit_generator(
  initial_state,
  rng_algorithm = "THREE_FRY",
  dtype,
  shape_out
)
```

## Arguments

- dtype:

  datatype of output

- shape_out:

  output shape

- initial:

  state seed

- rng_alogorithm:

  one of 'DEFAULT', 'THREE_FRY', 'PHILOX'
