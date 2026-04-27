# Create a Primitive

Builds an
[`AnvlPrimitive`](https://r-xla.github.io/anvl/reference/AnvlPrimitive.md)
metadata object, wraps `fn` with
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md), attaches the
metadata via `attr(., "primitive")`, prepends class `"JitPrimitive"`,
and (by default) registers the result under `name` in the primitive
registry.

The backend is always `"auto"` and cannot be configured.

## Usage

``` r
new_primitive(
  name,
  fn,
  subgraphs = character(),
  static = character(),
  device = NULL,
  register = TRUE
)
```

## Arguments

- name:

  (`character(1)`)  
  Primitive name.

- fn:

  (`function`)  
  Body of the primitive. Its formals become the formals of the returned
  JIT-compiled callable. Inside `fn`, the primitive is accessible via
  the lexically-bound symbol `self` (an
  [`AnvlPrimitive`](https://r-xla.github.io/anvl/reference/AnvlPrimitive.md));
  pass it as the first argument to
  [`graph_desc_add()`](https://r-xla.github.io/anvl/reference/graph_desc_add.md).

- subgraphs:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters that are subgraphs (for higher-order primitives).

- static:

  ([`character()`](https://rdrr.io/r/base/character.html) \|
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  Passed to [`jit()`](https://r-xla.github.io/anvl/reference/jit.md).

- device:

  (`NULL` \| `character(1)` \|
  [`device_arg()`](https://r-xla.github.io/anvl/reference/device_arg.md))  
  Passed to [`jit()`](https://r-xla.github.io/anvl/reference/jit.md).
  Useful for primitives with no array inputs (e.g. `prim_fill`) where
  the device must come from an explicit argument.

- register:

  (`logical(1)`)  
  If `TRUE` (default), register the result under `name` in the primitive
  registry.

## Value

A callable of class `c("JitPrimitive", "JitFunction")`.
