# anvil

Package website: [release](https://r-xla.github.io/anvil/) \|
[dev](https://r-xla.github.io/anvil/dev/)

Composable code transformation framework for R, allowing you to run
numerical programs at the speed of light. It currently implements JIT
compilation for very fast execution and reverse-mode automatic
differentiation. Programs can run on various hardware backends,
including CPU and GPU.

## Installation

{anvil} can be installed from GitHub or
[r-universe](https://r-xla.r-universe.dev/builds). During runtime, we
require `libprotobuf`. Source installation requires a `C++20` compiler
and `protoc` (protobuf compiler).

``` r
# Install release
# from r-universe (prebuilt binary)
install.packages("anvil", repos = c("https://cloud.r-project.org", "https://r-xla.r-universe.dev"))
# from GitHub (installation from source)
pak::pak("r-xla/anvil@*release")
# Install dev version
pak::pak("r-xla/anvil")
# Install CUDA support (linux x86_64): only requires a compatible driver
install.packages("cuda12.8", repos = "https://mlverse.r-universe.dev")
```

Prebuilt [Docker images](https://github.com/r-xla/docker) are also
available. See the *Installation* vignette on the package website for
more details.

## Quick Start

Below, we create a standard R function. We cannot directly call this
function, but first need to wrap it in a
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) call. If
the resulting function is then called on `AnvilArray`s – the primary
data type in {anvil} – it will be JIT compiled and subsequently
executed.

``` r
library(anvil)
f <- function(a, b, x) {
  a * x + b
}
f_jit <- jit(f)

a <- nv_scalar(1.0, "f32")
b <- nv_scalar(-2.0, "f32")
x <- nv_scalar(3.0, "f32")

f_jit(a, b, x)
#> AnvilArray
#>  1
#> [ CPUf32{} ]
```

Through automatic differentiation, we can also obtain the gradient of
the above function.

``` r
g_jit <- jit(gradient(f, wrt = c("a", "b")))
g_jit(a, b, x)
#> $a
#> AnvilArray
#>  3
#> [ CPUf32{} ] 
#> 
#> $b
#> AnvilArray
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

## When to use this package?

While {anvil} allows to run certain types of programs extremely fast, it
only applies to a certain category of problems. Specifically, it is
suitable for numerical algorithms, such as fitting bayesian models,
training neural networks or more generally numerical optimization.
Another restriction is that {anvil} needs to re-compile the code for
each new unique input shape. This has the advantage, that the compiler
can make memory optimizations, but the compilation overhead might be a
problem when inputs shapes are varying often and the programs run
quickly.

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
