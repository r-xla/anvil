# Make initial state

converts a random seed into a initial state tensor

## Usage

``` r
nv_seed2state(
  dtype = "ui64",
  shape_out = 2,
  random_seed = NULL,
  hash_algo = "sha512"
)
```

## Arguments

- dtype:

  output dtype either "ui32" or "ui64"

- shape_out:

  output shape

- random_seed:

  explicitly provide the random seed of a R session. auto-detects if not
  provided.

- hash_algo:

  hash algorithm to hash the random state with. Default is 'sha512'.
