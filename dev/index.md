# anvl

Package website: [release](https://r-xla.github.io/anvl/) \|
[dev](https://r-xla.github.io/anvl/dev/)

Composable code transformation framework for R, allowing you to run
numerical programs at the speed of light. It currently implements JIT
compilation for very fast execution and reverse-mode automatic
differentiation. Programs can run on various hardware backends,
including CPU and GPU.

## Installation

{anvl}[¹](#fn1) can be installed from GitHub or
[r-universe](https://r-xla.r-universe.dev/builds). During runtime, we
require `libprotobuf`. Source installation requires a `C++20` compiler
and `protoc` (protobuf compiler).

``` r
# Install release
# from r-universe (prebuilt binary)
install.packages("anvl", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
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

We define an R function operating on `AnvlArray`s – the primary data
type of {anvl}. It can be executed in either *eager* mode (each
operation is performed immediately) or *jit* mode (the whole function is
compiled into a single kernel via
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)).

``` r
library(anvl)
f <- function(a, b, x) {
  a * x + b
}

a <- nv_scalar(1.0, "f32")
b <- nv_scalar(-2.0, "f32")
x <- nv_scalar(3.0, "f32")

# Eager mode
f(a, b, x)
#> AnvlArray
#>  1
#> [ CPUf32{} ]

# JIT mode
f_jit <- jit(f)
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
- Eager and JIT mode:
  - Run operations eagerly for interactive use, or
    [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) entire
    functions into a single fused kernel for performance.
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
  - ✅ CPU backend is fully supported.
  - ✅ CUDA (NVIDIA GPU) backend is fully supported.
- **Linux (ARM)**
  - ✅ CPU backend is fully supported.
  - ❌ GPU is not supported.
- **Windows**
  - ✅ CPU backend is fully supported.
  - ⚠️ GPU is only supported via Windows Subsystem for Linux (WSL2).
- **macOS (ARM)**
  - ✅ CPU backend is supported.
  - ⚠️ Metal (Apple GPU) backend is available but not fully functional.
- **macOS (x86_64)**
  - ❌ Not supported.

## Acknowledgments

- This work is supported by [MaRDI](https://www.mardi4nfdi.de).
- The design of this package was inspired by and borrows from:
  - JAX, especially the [autodidax
    tutorial](https://docs.jax.dev/en/latest/autodidax.html).
  - The [microjax](https://github.com/joey00072/microjax) project.
- For JIT compilation, we leverage the [OpenXLA](https://openxla.org/)
  project.

------------------------------------------------------------------------

1.  A real anvil is a tool blacksmiths use to reshape metal; this
    package reshapes R code in a similar spirit. We spell the package
    name `anvl` (without the `i`) because `AnVIL` is already taken by a
    Bioconductor package.
