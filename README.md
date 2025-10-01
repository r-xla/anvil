
<!-- README.md is generated from README.Rmd. Please edit that file -->

# anvil

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

The goal of anvil is to provide a code transformation framework similar
to [jax](https://docs.jax.dev/en/latest/) for R. It currently provides
support for jit compilation and automatic differentiation.

## Installation

``` r
pak::pak("r-xla/anvil")
```

## Quick Start

Below, we create a simple R function, jit-compile and run it.
Afterwards, we also evaluate its gradient.

``` r
library(anvil)
f <- function(a, b, x) {
  a * x + b
}

a <- nv_scalar(1.0)
b <- nv_scalar(-2.0)
x <- nv_scalar(3.0)

fj <- jit(f)
fj(a, b, x)
#> PJRTBuffer<f32> 
#>  1.0000

g <- jit(gradient(f))

g(a, b, x)
#> [[1]]
#> PJRTBuffer<f32> 
#>  3.0000
#> 
#> [[2]]
#> PJRTBuffer<f32> 
#>  1.0000
#> 
#> [[3]]
#> PJRTBuffer<f32> 
#>  1.0000
```

## Features

- Automatic Differentiation:
  - Pullback for reverse mode automatic differentiation.
- Fast:
  - Code is JIT compiled into a single kernel.
  - Runs on various hardware, including CPU and GPU.
- Easy to contribute:
  - written almost entirely in R
- Extendable:
  - Easy to add new primitives and interpretation rules.

## Acknowledgments

- This work is supported by [MaRDI](https://www.mardi4nfdi.de).
- The design of this package was inspired by and borrows from:
  - JAX, especiall the [autodidax
    tutorial](https://docs.jax.dev/en/latest/autodidax.html).
  - The [microjax](https://github.com/joey00072/microjax) project.
- The project leverages [stableHLO](https://github.com/r-xla/stablehlo)
  and [pjrt](https://github.com/r-xla/pjrt) which in turn build upon the
  [OpenXLA](https://openxla.org/) project.
