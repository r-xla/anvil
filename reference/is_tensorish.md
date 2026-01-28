# Check if an Object is a Tensor-ish Object

Check if an object is a tensor-ish object.

## Usage

``` r
is_tensorish(x, literal = TRUE)
```

## Arguments

- x:

  (`any`)  
  Object to check.

- literal:

  (`logical(1)`)  
  Whether to allow R literals (i.e., `1L`, `1.0`, `TRUE`, etc.) to be
  considered tensor-ish. Defaults to `TRUE`.

## Value

`logical(1)`
