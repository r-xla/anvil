# XLA backend

Constructs the XLA backend, which stores array data in PJRT buffers (via
[`pjrt::pjrt_buffer()`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html))
and compiles jitted functions to XLA executables via
[`stablehlo()`](https://r-xla.github.io/anvil/dev/reference/stablehlo.md)
and
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).
This is the default backend.

## Usage

``` r
AnvilBackendXla()
```

## Value

An
[`AnvilBackend`](https://r-xla.github.io/anvil/dev/reference/AnvilBackend.md)
object with subclass `"AnvilBackendXla"`.

## Data representation

An
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
with `backend = "xla"` wraps a
[`pjrt::pjrt_buffer()`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html)
stored in the `$data` field. The buffer owns the memory holding the
tensor values and may live on any device supported by PJRT (CPU, CUDA,
Metal, ...). Calling
[`as_array()`](https://r-xla.github.io/anvil/dev/reference/as_array.md)
transfers the buffer contents back to an R array; calling
[`nv_array()`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
on an R object uploads it to the requested device.

Each `AnvilArray` therefore has an associated device, queryable via
[`device()`](https://r-xla.github.io/anvil/dev/reference/device.md). A
device is a
[`pjrt::as_pjrt_device()`](https://r-xla.github.io/pjrt/reference/as_pjrt_device.html)
object (e.g. the platform `"cpu"` or `"cuda"`, optionally with an index
such as `"cuda:1"`). When `device` is `NULL` in
[`nv_array()`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
or the [`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)
wrapper, the device defaults to the `PJRT_PLATFORM` environment variable
(falling back to `"cpu"`), or is inferred from the existing inputs of a
jitted call. Operations require all inputs to live on the same device.

## XLA JIT arguments

- `donate` ([`character()`](https://rdrr.io/r/base/character.html),
  default [`character()`](https://rdrr.io/r/base/character.html)): names
  of arguments whose underlying buffers may be donated to (i.e.,
  reused/consumed by) the compiled XLA executable. Donated buffers must
  not be used again by the caller after the call; this can reduce memory
  usage and copies for large inputs. Must not overlap with `static`.

## See also

[`AnvilBackend()`](https://r-xla.github.io/anvil/dev/reference/AnvilBackend.md),
[`AnvilBackendQuickr()`](https://r-xla.github.io/anvil/dev/reference/AnvilBackendQuickr.md),
[`local_backend()`](https://r-xla.github.io/anvil/dev/reference/local_backend.md),
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md).
