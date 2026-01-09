# Matrix Multiplication

Matrix multiplication of two tensors.

## Usage

``` r
nv_matmul(lhs, rhs)
```

## Arguments

- lhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))

- rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

- `lhs`: `(b1, ..., bk, m, n)`

- `rhs`: `(b1, ..., bk, n, p)`

- output: `(b1, ..., bk, m, p)`
