# Matrix Multiplication

Matrix multiplication of two tensors.

## Usage

``` r
nv_matmul(lhs, rhs)
```

## Arguments

- lhs:

  ([`tensorish`](tensorish.md))

- rhs:

  ([`tensorish`](tensorish.md))

## Value

[`tensorish`](tensorish.md)

## Shapes

- `lhs`: `(b1, ..., bk, m, n)`

- `rhs`: `(b1, ..., bk, n, p)`

- output: `(b1, ..., bk, m, p)`

## Broadcasting

All dimensions but the last two are broadcasted.
