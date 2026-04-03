# Installation

Currently, {anvil} is not available on CRAN, so you either have to
install it via r-universe or from GitHub.

## System Dependencies

You need a C++20 compiler, `libprotobuf`, and the `protobuf-compiler`.

## Installing the latest release from GitHub

You can install the latest GitHub release via:

``` r
pak::pak("r-xla/anvil@*release")
```

You can install the latest release from r-universe via:

``` r
install.packages("anvil", repos = "https://r-xla.r-universe.dev")
```

To confirm that your CPU installation is working, run:

``` r
library(anvil)
nv_scalar(1, device = "cpu")
```

## Installation the development version from GitHub

The development version can be installed via:

``` r
pak::pak("r-xla/anvil")
```

## GPU Support

Running {anvil} with GPU support only works on Linux (amd64/x86-64) or
via WSL2 on Windows. In principle, MPS support on macOS is available but
many features are missing and the PJRT plugin provided by Apple is
heavily outdated.

### CUDA

To use the CUDA backend on Linux install the {cuda12.8} R package which
provides the required CUDA runtime libraries and you only need to have a
compatible CUDA driver.

``` r
pak::pak("mlverse/cudatoolkit/cuda12.8")
```

Alternatively, install from
[r-universe](https://mlverse.r-universe.dev/).

``` r
install.packages("cuda12.8", repos = "https://mlverse.r-universe.dev")
```

When the {cuda12.8} package is not installed, the correct runtime
libraries need to be installed on the system and discoberable via
`LD_LIBRARY_PATH`. This can be difficult to set up. The specific
versions of the CUDA runtime libraries provided with {cuda12.8} are
provided
[here](https://github.com/mlverse/cudatoolkit/blob/main/cuda12.8/inst/components.tsv).

**Troubleshooting**

To trouble-shoot the CUDA installation, run the following in a new R
session for maximum debug output.

``` r
Sys.setenv(PJRT_DEBUG = "1", TF_CPP_MIN_LOG_LEVEL = "0")
anvil::nv_scalar(1, device = "cuda")
```

Note that if another package is using a different cudatoolkit package
(e.g. when using {torch}), there might be some issues, so in this case
it’s best to run separate R processes.

If you are working on a server, you can also use the prebuilt Docker
images, see the next section.

## Docker

Prebuilt Docker images are available in
[r-xla/docker](https://github.com/r-xla/docker). This includes a CUDA
and CPU build for amd64/x86-64 architecture:

### Available Images

| Image        | Description                          |
|--------------|--------------------------------------|
| `anvil-cpu`  | CPU support, based on `rocker/r-ver` |
| `anvil-cuda` | GPU support with CUDA 12.8           |

Note that running the GPU container requires the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to be installed on the host.

### Tags

Each image is available with two tags:

| Tag        | Description                                  |
|------------|----------------------------------------------|
| `:latest`  | Built from the `main` branch (rebuilt daily) |
| `:release` | Built from the latest release                |
