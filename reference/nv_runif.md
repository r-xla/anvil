# Internal: Random Unit Uniform Numbers

generate random uniform numbers in \[0, 1)

generate random uniform numbers in \]lower, upper\[

## Usage

``` r
nv_unif_rand(initial_state, dtype = "f64", shape_out)

nv_runif(initial_state, dtype = "f64", shape_out, lower = 0, upper = 1)
```

## Arguments

- initial_state:

  state seed

- dtype:

  output dtype either "f32" or "f64"

- shape_out:

  output shape

- lower:

  lower bound

- upper:

  upper bound
