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
options(repos = c(
  rxla = "https://r-xla.r-universe.dev",
  CRAN = "https://cloud.r-project.org/"
))
install.packages("anvil")
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

Running {anvil} with GPU support only works on Linux or via WSL2 on
Windows. In principle, MPS support on macOS is available but many
features are missing and the PJRT plugin is heavily outdated. To use
{anvil} with GPU support, you need to have CUDA installed. The PJRT CUDA
plugin was built with the following versions:

- CUDA 12.8.1
- CUDNN 9.8.0
- NVSHMEM 3.2.5

Besides, there is [this
dockerfile](https://github.com/r-xla/docker/blob/main/cuda-base/Dockerfile)
which is used in the GitHub Actions CI, and the versions used therein
should also work.

To confirm that your GPU installation is working, run:

``` r
library(anvil)
nv_scalar(1, device = "cuda")
```

If you are working on a server, you can also use the prebuilt Docker
images, see the next section.

## Docker

Prebuilt Docker images are available in
[r-xla/docker](https://github.com/r-xla/docker). This includes a CUDA
and CPU build for amd64/x86-64 architecture:

### Available Images

| Image             | Description                                           |
|-------------------|-------------------------------------------------------|
| `anvil-cpu`       | CPU support, based on `rocker/r-ver`                  |
| `anvil-cuda`      | GPU support with CUDA 12.8                            |
| `anvil-cuda-base` | Base image for `anvil-cuda` without `anvil` installed |

Note that running the GPU container requires the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to be installed on the host.

### Tags

Each image is available with two tags:

| Tag        | Description                                  |
|------------|----------------------------------------------|
| `:latest`  | Built from the `main` branch (rebuilt daily) |
| `:release` | Built from the latest release                |
