# Update Subset

Updates elements of a tensor at specified positions. See
vignette("subsetting") for more details.

## Usage

``` r
# S3 method for class 'AnvilBox'
x[...] <- value

# S3 method for class 'AnvilTensor'
x[...] <- value

nv_subset_assign(x, ..., value)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Input tensor to update.

- ...:

  Subset specifications.

- value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Values to write. Scalars are broadcast to the subset shape. Non-scalar
  values must have a shape matching the subset shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md) A
new tensor with the subset updated.

## See also

[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md)
