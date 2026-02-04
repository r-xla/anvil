# Convert to Abstract Tensor

Convert an object to its abstract tensor representation
([`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)).

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
  Whether to convert to a pure `AbstractTensor` and not e.g.
  `LiteralTensor` or `ConcreteTensor`.

## Value

[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)

## Examples

``` r
# R literals become LiteralTensors (ambiguous by default, except logicals)
to_abstract(1.5)
#> LiteralTensor(1.5, f32?, ()) 
to_abstract(1L)
#> LiteralTensor(1, i32?, ()) 
to_abstract(TRUE)
#> LiteralTensor(TRUE, i1, ()) 

# AnvilTensors become ConcreteTensors
to_abstract(nv_tensor(1:4))
#> ConcreteTensor
#>  1
#>  2
#>  3
#>  4
#> [ CPUi32{4} ] 

# Use pure = TRUE to strip subclass info
to_abstract(nv_tensor(1:4), pure = TRUE)
#> AbstractTensor(dtype=i32, shape=4) 
```
