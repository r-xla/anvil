# jit(): concrete backend + device_arg

## Motivation

`jit()` currently supports two compatible ways to request a dynamic (call-time) device:

1. `backend = "auto"` + `device = device_arg("x")` — routes through `jit_auto()`,
   which uses `backend()` dispatch on the value passed for `x` to pick a
   per-backend jitted implementation.
2. A concrete `backend = "xla"` (or `"quickr"`) + a static/explicit `device`
   argument — standard path through `jit_with_backend()`.

A `TODO` in `R/jit.R:77` marks the missing third combination: concrete backend
plus `device_arg(...)`. This is useful when the user has already committed to
a backend but wants the device to come from a runtime argument, e.g.

```r
f <- jit(
  \(x) nv_scalar(1, device = x),
  backend = "xla",
  device = device_arg("x")
)
f("cuda")                           # string accepted — backend is known
f(pjrt::pjrt_device("cpu"))         # device object accepted
f(nv_device("cpu", "quickr"))       # error: wrong backend
```

Because the backend is fixed at jit time, strings ("cpu", "cuda", ...) can be
resolved without needing `backend()` dispatch.

## Design

### Entry point: `R/jit.R`

Replace the `TODO` in the `is_device_arg(device)` branch with:

```r
if (is_device_arg(device)) {
  backend <- backend %||% default_backend()
  if (backend == "auto") {
    return(jit_auto(f, static, cache_size, device_argname = device$argname, ...))
  }
  assert_subset(device$argname, formalArgs2(f))
  static <- unique(c(static, device$argname))
  return(jit_with_backend(f, static, cache_size, backend, device = device, ...))
}
```

The referenced argument is marked static (same rule as the `jit_auto` branch
already applies). The `device_arg` is forwarded into
`jit_with_backend(...)` so the backend-specific jit impl receives it.

### XLA backend: `R/backend-xla.R`

`jit_prepare_call()` already handles a `device_arg` correctly: it extracts
`args[[device$argname]]`, runs it through `nv_device(value, backend = "xla")`,
and stores the normalized device as `prep$device`. The `nv_device()` call is
the single chokepoint that enforces backend matching — passing a `QuickrDevice`
produces a clear error.

Two adjustments in `jit_xla_impl()`:

1. Compute a concrete device for downstream use:
   ```r
   device_concrete <- if (is_device_arg(device)) prep$device else device
   ```
2. Use `device_concrete` in place of `device` for:
   - `cache_key` (keyed on in_tree, avals_in, and device)
   - the `device =` argument to `compile_xla()`

Without step 2 on `compile_xla()`, 0-input functions (no dynamic array args,
which is the primary use case for `device_arg`) would fall back to
`default_device("xla")` during compilation since the device-providing argument
is static and not present in `arg_devices`.

Cache correctness note: when `device_arg` is used, the device value lives in
a static argument position, so `avals_in` (which passes static values through
unchanged) already captures device variation between calls. Distinct call-time
devices therefore produce distinct cache entries regardless of the third
`cache_key` element.

### Quickr backend: `R/backend-quickr.R`

Currently `jit_quickr_impl()` ignores the `device` parameter it receives and
does not pass it into `jit_prepare_call()`. For symmetry, thread `device` into
`jit_prepare_call(..., device = device, backend = "quickr")` so that when
`device` is a `device_arg` the user-supplied value goes through
`nv_device(value, backend = "quickr")` and produces the same
backend-mismatch error that the XLA path produces.

No other quickr-side changes are needed — quickr has only one device, but the
user-facing validation (string or matching-backend device object) is the
observable contract and should behave consistently.

### User-facing behavior after change

| user-supplied value for the device arg    | xla    | quickr |
|-------------------------------------------|--------|--------|
| string (e.g. `"cpu"`, `"cuda"`)           | OK     | OK (`"cpu"` only) |
| matching-backend device object            | OK     | OK     |
| other-backend device object               | error  | error  |
| `NULL`                                    | falls back to `PJRT_PLATFORM` default (pre-existing behavior in `jit_prepare_call`) | same |

## Tests (`tests/testthat/test-jit.R`)

Replace the existing

```r
it("checks backend for device_arg", {
  expect_error(
    jit(function(x, dev) x, device = device_arg("dev"), backend = "xla")
  )
})
```

with three tests (keeping them inside the existing
`describe("jit: backend and device combinations", …)` block):

```r
it("concrete backend + device_arg: device as string", {
  f <- jit(
    \(x) nv_scalar(1, device = x),
    backend = "xla",
    device = device_arg("x")
  )
  expect_equal(device(f("cpu")), nv_device("cpu", "xla"))
})

it("concrete backend + device_arg: device as device object", {
  f <- jit(
    \(x) nv_scalar(1, device = x),
    backend = "xla",
    device = device_arg("x")
  )
  dev <- nv_device("cpu", "xla")
  expect_equal(device(f(dev)), dev)
})

it("concrete backend + device_arg: wrong backend device errors", {
  skip_if_not_installed("quickr")
  f <- jit(
    \(x) nv_scalar(1, device = x),
    backend = "xla",
    device = device_arg("x")
  )
  expect_error(f(nv_device("cpu", "quickr")), "backend")
})
```

## Non-goals

- No change to the `backend = "auto"` + `device_arg` path.
- No changes to cache semantics beyond what is needed for the new case. In
  particular, two calls with the "same logical device" supplied as a string
  vs a device object still create distinct cache entries; this matches the
  pre-existing static-arg equality model.
- No new user-facing API.
