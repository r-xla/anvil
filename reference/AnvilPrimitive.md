# AnvilPrimitive

Primitive interpretation rule. Note that `[[` and `[[<-` access the
interpretation rules. To access other fields, use `$` and `$<-`.

## Usage

``` r
AnvilPrimitive(name, higher_order = FALSE, subgraphs = character())
```

## Arguments

- name:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The name of the primitive.

- higher_order:

  (`logical(1)`)  
  Whether the primitive is higher-order (contains subgraphs). Default is
  `FALSE`.

- subgraphs:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters that are subgraphs. Only used if
  `higher_order = TRUE`.

## Value

(`AnvilPrimitive`)
