# anvl

Package website: [release](https://r-xla.github.io/anvl/) \|
[dev](https://r-xla.github.io/anvl/dev/)

Accelerated array computing and code transformations for R, allowing you
to run numerical programs at the speed of light. The package supports
JIT compilation for very fast execution and reverse-mode automatic
differentiation. Programs can run on CPU and NVIDIA GPU.

## Installation

``` r

install.packages("anvl", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
```

Install CUDA support on linux amd64:

``` r

install.packages("cuda12.8", repos = "https://mlverse.r-universe.dev")
```

See the [Installation
guide](https://r-xla.github.io/anvl/articles/installation.html) for more
details, including prebuilt Docker images.

## Why anvl

anvl makes numerical R code run fast on CPUs and GPUs, and computes
gradients of your functions automatically. It serves the same role for R
that JAX serves for Python.

There are three ideas:

- **Compilation.** Wrap any R function in
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md). anvl
  converts the R function into an optimized program via XLA – the same
  compiler that powers JAX and TensorFlow.
- **Function transformation.** Programmatically derive new functions
  from existing ones. Currently the only available transformation is
  reverse-mode automatic differentiation via
  [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md),
  which returns the derivative of a function as another R function.
- **Hardware portability.** The same code runs on CPU or NVIDIA CUDA
  GPU.

Compared to R [`torch`](https://torch.mlverse.org), anvl is built around
compilation:
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) sees the
*whole function* at once and compiles it. The resulting programs can
therefore be considerably faster and custom operations can be
implemented more easily.

## Quick Start

We define an R function operating on `AnvlArray`s – the primary data
type of {anvl}. It can be executed in either *eager* mode (each
operation is performed immediately) or *jit* mode (the whole function is
compiled into a single executable via
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

- **JIT and eager modes** – compile whole functions with
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) for
  performance, or run operations eagerly for interactive use and
  debugging.
- **Reverse-mode autodiff** –
  [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
  and
  [`value_and_gradient()`](https://r-xla.github.io/anvl/dev/reference/value_and_gradient.md)
  for functions with scalar outputs.
- **Asynchronous execution** – compiled calls return immediately and run
  on the accelerator while R prepares the next call, keeping CPU and
  device busy in parallel.
- **Extensible** – the package is written almost entirely in R; new
  primitives and transformations can be added without leaving R.
- **Multiple backends** – compiled programs run via XLA on CPU or GPU,
  with an experimental
  [`quickr`](https://CRAN.R-project.org/package=quickr)-based Fortran
  backend for CPU-only workloads.

## Platform Support

| Platform              | CPU |        GPU         |
|-----------------------|:---:|:------------------:|
| Linux (x86_64)        |  ✓  |       ✓ CUDA       |
| Linux (ARM)           |  ✓  |         ✗          |
| Windows               |  ✓  | ◐ WSL2 only (CUDA) |
| macOS (Apple Silicon) |  ✓  |         ✗          |
| macOS (Intel)         |  ✗  |         ✗          |

✓ fully supported  ·  ◐ limited support  ·  ✗ not supported

## Acknowledgments

- This work is supported by [MaRDI](https://www.mardi4nfdi.de).
- The design of this package was inspired by and borrows from:
  - JAX, especially the [autodidax
    tutorial](https://docs.jax.dev/en/latest/autodidax.html).
  - The [microjax](https://github.com/joey00072/microjax) project.
- For JIT compilation, we leverage the [OpenXLA](https://openxla.org/)
  project.
