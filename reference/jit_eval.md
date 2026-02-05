# Jit an Evaluate an Expression

Compiles and evaluates an expression.

## Usage

``` r
jit_eval(expr, device = NULL)
```

## Arguments

- expr:

  (`expression`)  
  Expression to run.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device to use. By default (`NULL`), the device is inferred from
  the tensors encountered during tracing, falling back to
  `PJRT_PLATFORM` or `"cpu"`.

## Value

(`any`)  
Result of the expression.
