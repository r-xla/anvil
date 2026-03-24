# Compare AbstractArray Types

Compare two abstract arrays for type equality.

## Usage

``` r
eq_type(e1, e2, ambiguity)

neq_type(e1, e2, ambiguity)
```

## Arguments

- e1:

  ([`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md))  
  First array to compare.

- e2:

  ([`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md))  
  Second array to compare.

- ambiguity:

  (`logical(1)`)  
  Whether to consider the ambiguous field when comparing. If `TRUE`,
  arrays with different ambiguity are not equal. If `FALSE`, only dtype
  and shape are compared.

## Value

`logical(1)` - `TRUE` if the arrays are equal, `FALSE` otherwise.

## Examples

``` r
a <- nv_abstract("f32", c(2L, 3L))
b <- nv_abstract("f32", c(2L, 3L))

# Same dtype and shape
eq_type(a, b, ambiguity = FALSE)
#> [1] TRUE

# Different dtype
eq_type(a, nv_abstract("i32", c(2L, 3L)), ambiguity = FALSE)
#> [1] FALSE

# Different shape
eq_type(a, nv_abstract("f32", c(3L, 2L)), ambiguity = FALSE)
#> [1] FALSE

# ambiguity parameter controls whether ambiguous field is compared
c <- nv_abstract("f32", c(2L, 3L), ambiguous = TRUE)
eq_type(a, c, ambiguity = FALSE)
#> [1] TRUE
eq_type(a, c, ambiguity = TRUE)
#> [1] FALSE

# neq_type is the negation of eq_type
neq_type(a, b, ambiguity = FALSE)
#> [1] FALSE
```
