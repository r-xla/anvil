# Matrix Multiplication

Matrix multiplication of two tensors.

## Usage

``` r
nv_matmul(lhs, rhs)
```

## Arguments

- lhs:

  ([`nv_tensor`](nv_tensor.md))

- rhs:

  ([`nv_tensor`](nv_tensor.md))

## Value

[`nv_tensor`](nv_tensor.md)

## Shapes

- `lhs`: `(b1, ..., bk, m, n)`

- `rhs`: `(b1, ..., bk, n, p)`

- output: `(b1, ..., bk, m, p)`

## Broadcasting

All dimensions but the last two are broadcasted.
