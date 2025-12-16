# Type Promotion Rules

Computes the common dtype for a set of abstract types, respecting
whether a type is ambiguous or not. A type is ambiguous if it comes from
a literal (like 1 or 1.0) or was promoted to an ambiguous type.
Promoting to an ambiguous type can happen in scenarios like `x + 1.2`,
where `x` is a bool or an int.

## Usage

``` r
common_dtype(
  lhs_dtype,
  rhs_dtype,
  lhs_ambiguous = FALSE,
  rhs_ambiguous = FALSE
)
```

## Arguments

- lhs_dtype:

  ([`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  The left-hand side type.

- rhs_dtype:

  ([`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  The right-hand side type.

- lhs_ambiguous:

  (`logical(1)`)  
  Whether the left-hand side type is ambiguous.

- rhs_ambiguous:

  (`logical(1)`)  
  Whether the right-hand side type is ambiguous.

## Value

(`list(dtype = [`stablehlo::TensorDataType`], ambiguous = `logical(1)\`)  
