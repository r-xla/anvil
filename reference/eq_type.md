# Compare AbstractTensor Types

Compare two AbstractTensors for type equality.

## Usage

``` r
eq_type(e1, e2, ambiguity)

neq_type(e1, e2, ambiguity)
```

## Arguments

- e1:

  ([`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md))  
  First tensor to compare.

- e2:

  ([`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md))  
  Second tensor to compare.

- ambiguity:

  (`logical(1)`)  
  Whether to consider the ambiguous field when comparing. If `TRUE`,
  tensors with different ambiguity are not equal. If `FALSE`, only dtype
  and shape are compared.

## Value

`logical(1)` - `TRUE` if the tensors are equal, `FALSE` otherwise.
