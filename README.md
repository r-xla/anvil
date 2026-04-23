
<!-- README.md is generated from README.Rmd. Please edit that file -->

# anvl <img src="man/figures/logo.png" align="right" width = "120" />

Package website: [release](https://r-xla.github.io/anvl/) \|
[dev](https://r-xla.github.io/anvl/dev/)

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
![R-CMD-check](https://github.com/r-xla/anvl/actions/workflows/R-CMD-check.yaml/badge.svg)
[![CRAN
status](https://www.r-pkg.org/badges/version/anvl)](https://CRAN.R-project.org/package=anvl)
[![codecov](https://codecov.io/gh/r-xla/anvl/branch/main/graph/badge.svg)](https://codecov.io/gh/r-xla/anvl)
[![r-universe](https://r-xla.r-universe.dev/badges/anvl)](https://r-xla.r-universe.dev/anvl)
![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)
<!-- badges: end -->

Composable code transformation framework for R, allowing you to run
numerical programs at the speed of light. It currently implements JIT
compilation for very fast execution and reverse-mode automatic
differentiation. Programs can run on various hardware backends,
including CPU and GPU.

## Installation

{anvl}[^1] can be installed from GitHub or
[r-universe](https://r-xla.r-universe.dev/builds). During runtime, we
require `libprotobuf`. Source installation requires a `C++20` compiler
and `protoc` (protobuf compiler).

``` r
# Install release
# from r-universe (prebuilt binary)
install.packages("anvl", repos = c("https://cloud.r-project.org", "https://r-xla.r-universe.dev"))
# from GitHub (installation from source)
pak::pak("r-xla/anvl@*release")
# Install dev version
pak::pak("r-xla/anvl")
# Install CUDA support (linux x86_64): only requires a compatible driver
install.packages("cuda12.8", repos = "https://mlverse.r-universe.dev")
```

Prebuilt [Docker images](https://github.com/r-xla/docker) are also
available. See the *Installation* vignette on the package website for
more details.

## Quick Start

Below, we create an R function and `jit()` it. Then, we call the
resulting function on `AnvlArray`s, which will compile and subsequently
execute it.

``` r
library(anvl)
f <- function(a, b, x) {
  a * x + b
}
f_jit <- jit(f)

a <- nv_scalar(1.0, "f32")
b <- nv_scalar(-2.0, "f32")
x <- nv_scalar(3.0, "f32")

f_jit(a, b, x)
#> AnvlArray
#>  1
#> [ CPUf32{} ]
```

Through automatic differentiation, we can also obtain the gradient of
the above function.

``` r
g_jit <- jit(gradient(f, wrt = c("a", "b")))
g_jit(a, b, x)
#> $a
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
#> 
#> $b
#> AnvlArray
#>  1
#> [ CPUf32{} ]
```

For more complex examples, such as implementing a Gaussian Process, see
the package website.

## Main Features

- Automatic Differentiation:
  - Gradients for functions with scalar outputs are supported.
- Fast:
  - Code is JIT compiled into a single kernel.
  - Runs on different hardware backends, including CPU and GPU.
  - Asyncronous allocation and execution, allowing the accelerator to do
    it’s job while R interpretes.
- Extendable:
  - It is possible to add new primitives, transformations, and (with
    some effort) new backends.
  - The package is written almost entirely in R.
- Multi-backend:
  - The backend supports execution via XLA as well as an experimental
    {quickr}-based Fortran backend (CPU only).

## Platform Support

- **Linux (x86_64)**
  - :white_check_mark: CPU backend is fully supported.
  - :white_check_mark: CUDA (NVIDIA GPU) backend is fully supported.
- **Linux (ARM)**
  - :white_check_mark: CPU backend is fully supported.
  - :x: GPU is not supported.
- **Windows**
  - :white_check_mark: CPU backend is fully supported.
  - :warning: GPU is only supported via Windows Subsystem for Linux
    (WSL2).
- **macOS (ARM)**
  - :white_check_mark: CPU backend is supported.
  - :warning: Metal (Apple GPU) backend is available but not fully
    functional.
- **macOS (x86_64)**
  - :x: Not supported.

## Acknowledgments

- This work is supported by [MaRDI](https://www.mardi4nfdi.de).
- The design of this package was inspired by and borrows from:
  - JAX, especially the [autodidax
    tutorial](https://docs.jax.dev/en/latest/autodidax.html).
  - The [microjax](https://github.com/joey00072/microjax) project.
- For JIT compilation, we leverage the [OpenXLA](https://openxla.org/)
  project.

[^1]: A real anvil is a tool blacksmiths use to reshape metal; this
    package reshapes R code in a similar spirit. We spell the package
    name `anvl` (without the `i`) because `AnVIL` is already taken by a
    Bioconductor package.
