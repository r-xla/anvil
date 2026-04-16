# Convert to Abstract Array

Convert an object to its abstract array representation
([`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md)).

## Usage

``` r
to_abstract(x, pure = FALSE)
```

## Arguments

- x:

  (`any`)  
  Object to convert.

- pure:

  (`logical(1)`)  
  Whether to convert to a pure `AbstractArray` and not e.g.
  `LiteralArray` or `ConcreteArray`.

## Value

[`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
# R literals become LiteralArrays (ambiguous by default, except logicals)
to_abstract(1.5)
to_abstract(1L)
to_abstract(TRUE)

# AnvilArrays become ConcreteArrays
to_abstract(nv_array(1:4))

# Use pure = TRUE to strip subclass info
to_abstract(nv_array(1:4), pure = TRUE)
}
```
