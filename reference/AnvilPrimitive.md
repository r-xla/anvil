# AnvilPrimitive

Primitive interpretation rule. Note that `[[` and `[[<-` access the
interpretation rules. To access other fields, use `$` and `$<-`.

A primitive is considered higher-order if it contains subgraphs.

## Usage

``` r
AnvilPrimitive(name, subgraphs = character())
```

## Arguments

- name:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The name of the primitive.

- subgraphs:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters that are subgraphs. Only used if
  `higher_order = TRUE`.

## Value

(`AnvilPrimitive`)
