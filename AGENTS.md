## Project Goal

I am developing a deep-learning framework that is inspired by the jax autodidax tutorial, which is located in ~/r-xla/anvil/autodidax.ipynb.
I also attached the "microjax" tutorial that is located in ~/r-xla/anvil/microjax.py which is more minimal and better to understand, but which does not cover jit-compilation.

# Project Overview

The package consists of:
I want to deviate from the JAX tutorial in various ways:

Restricted Feature Set:

For now, the goal of the project is only to support the following transformations

1. JIT-compilation of R functions to StableHLO programs
2. Differentiation (backward) of scalar R functions

## User Facing API

Defining the function:

```r
f <- function(lhs, rhs) {
  a <- prim_add(lhs, rhs)
  b <- prim_mul(a, nvl_array(2.0))
  return(b)
}
```

Staging out:

```r
ir <- stage_out(f,
  ShapedArray(dtype = "f32", shape = 1L),
  ShapedArray(dtype = "f32", shape = 1L)
)
```

Jit compilation:

```r
fcomp <- compile(ir)

fcomp(
  nvl_array(1.0),
  nvl_array(2.0)
)
#> nvl_array(6.0)
```

Differentiation:

```r
ir_deriv <- grad(ir,
  ShapedArray(dtype = "f32", shape = 1L),
  ShapedArray(dtype = "f32", shape = 1L)
)

fderiv <- compile(ir_deriv),

fderiv(
  nvl_array(1.0),
  nvl_array(2.0)
)
#> nvl_array(2.0)
```

## Design Notes

Instead of using 'bind' like in JAX, I want to use S7 generics to handle everything, as they
provide a natural way to handle dispatching.

To do so, I will define S7 generics like `prim_add`, `prim_mul`, `prim_neg`, etc. like seen above.

```r
prim_add <- S7::new_generic("prim_add", c("lhs", "rhs"), function(lhs, rhs) {
  S7::S7_dispatch()
})
```

We now need to define methods for:

1. Staging out to an Intermediate Representation (IR) that is similar to JAXPRs.
   This IR is also defined in S7 and in the file R/ir.R of the anvil package.
2. Backward pass. This will take an IR and return a new IR, which will represent the
   gradient of the original function.
   In microjax.py this is defined as backward_pass.
3. Jit compilation, which will convert an IR to a stableHLO program.
   The stablehlo package is located in ~/r-xla/stablehlo
