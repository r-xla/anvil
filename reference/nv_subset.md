# Subset a Tensor

Extracts a subset from a tensor. See vignette("subsetting") for more
details.

## Usage

``` r
# S3 method for class 'AnvilBox'
x[...]

# S3 method for class 'AnvilTensor'
x[...]

nv_subset(x, ...)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Input tensor to subset.

- ...:

  Subset specifications.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
