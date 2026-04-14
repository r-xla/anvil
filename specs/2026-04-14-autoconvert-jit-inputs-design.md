# Autoconvert inputs at JIT/XLA call boundary

**Date:** 2026-04-14
**Status:** Proposed

## Goal

Today, calling a `jit()`-wrapped or `xla()`-compiled function with a plain
R scalar or array errors with `"Expected AnvilArray, but got <class>"`. The
user has to wrap every input in `nv_array()` / `nv_scalar()` themselves.

Autoconvert trivially-convertible inputs at the call boundary so the
common case "just works", while still erroring clearly for inputs that
cannot be unambiguously converted.

## Conversion rule

Applied per non-static leaf of the input tree:

| Input                                      | Action                                     |
| ------------------------------------------ | ------------------------------------------ |
| `is_anvil_array(x)`                        | Pass through                               |
| atomic, `length(x) == 1`, `is.null(dim(x))` | `nv_scalar(x, backend = backend)` (shape `()`) |
| `is.array(x)` (has `dim`)                  | `nv_array(x, backend = backend)` (shape from `dim`) |
| anything else                              | Error: must be an `AnvilArray`, scalar, or array |

Notable consequences:

- A bare vector `c(1, 2, 3)` does **not** autoconvert (it is not
  `is.array()`). The user must pass `array(c(1, 2, 3))` or
  `nv_array(c(1, 2, 3))` explicitly. This keeps the rule unambiguous: the
  shape of a bare vector is otherwise underspecified (shape `(3)` vs
  coercion to `(1, 3)` is a judgment call).
- `NULL`, lists, functions, environments, etc. still error — the rule only
  fires for concrete scalars and arrays.
- Static leaves are never touched regardless of type.

## Where the conversion happens

### Shared helper

A new function `autoconvert_input(x, backend)` in `R/jit.R` encapsulates
the rule. It takes the target backend name so the created array lives on
the right backend (matches the backend of the `JitFunction` being
invoked; always `"xla"` for `xla()`).

### `jit()` path

`jit_prepare_call()` gains a `backend` parameter. After it flattens the
call args, it applies `autoconvert_input` to each non-static leaf. `args`
is then rebuilt via `unflatten(in_tree, args_flat)` so the structured and
flat views stay consistent.

Both `jit_xla_impl()` and `jit_quickr_impl()` pass their backend name
through to `jit_prepare_call()`.

The `currently_tracing()` branches of both impls are **unchanged** — when
a jit is invoked inside another jit's trace, inputs are already
`GraphValue`s from the outer trace, and autoconverting would be wrong.

### `xla()` path

`xla()` autoconverts top-level args directly (no flatten/unflatten
needed; it already assumes flat top-level args). Backend is always
`"xla"`. The conversion happens between the `eval` step and the
`a$data` unwrap.

## Quickr: flat-args internal path

Orthogonal cleanup enabled by the same refactor: `jit_quickr_impl`
currently passes the structured `prep$args` to the compiled quickr
function, whose wrapper then re-flattens via `flatten()`. Since
`jit_prepare_call()` has already produced `args_flat`, flattening twice
is wasted work.

### Change

`compile_to_quickr()` additionally returns a flat-args entry point
(call it `fun_flat`) that:

1. takes `args_flat` (including static leaves) and `is_static_flat`
2. performs the static-argument identity check (same semantics as the
   current wrapper)
3. strips static leaves, prepends const args, and calls the inner
   quickr-compiled function with named leaf args
4. restores the output tree

`jit_quickr_impl` uses `fun_flat` with flat args already unwrapped from
`AnvilArray`s (via `as_array()` per leaf), and stores `fun_flat` in the
LRU cache.

The existing structured wrapper returned by `graph_to_quickr_function()`
stays unchanged for users of the public low-level API.

### Factoring

Pull the "call inner quickr fn with flat leaf args" body (the static
check, const prepending, `do.call(inner, …)`, and output restoration)
out of `graph_to_quickr_make_wrapper()` into a helper used by both the
structured wrapper and `fun_flat`. This keeps the two paths in sync.

## Error messages

The new error for non-convertible inputs is raised inside
`autoconvert_input()` with a class and value preview:

> Cannot autoconvert input to an `AnvilArray`: expected an `AnvilArray`, a
> length-1 atomic (scalar), or an `is.array()` value, but got
> `<cls>` of length `<n>`.

The backend-specific backend-mismatch errors (`backend(x) != "xla"` /
`!= "quickr"`) remain unchanged; they now fire only when the input was
already an `AnvilArray` on the wrong backend.

## Testing

New tests (in `tests/testthat/test-jit.R` or a new `test-autoconvert.R`):

1. `jit(\(x) x + 1)(1)` works; result is shape `()`.
2. `jit(\(x) x)(matrix(1:4, 2, 2))` works; result is shape `(2, 2)`.
3. `jit(\(x) x)(c(1, 2, 3))` errors.
4. Static args are not converted:
   `jit(\(x, flag) ..., static = "flag")(nv_array(1), TRUE)` behaves as
   before; passing a non-AnvilArray non-scalar as `flag` is still fine.
5. `xla()` with a scalar and a matrix input works.
6. Same cases repeated for the quickr backend (gated on
   `skip_if_not_installed("quickr")`).
7. Quickr with a nested input tree (e.g. `f(list(x, y))`) still produces
   the correct result through the flat path. No-double-flatten is an
   internal invariant, not a public behaviour — it is covered by the
   existing quickr tests still passing.

## Non-goals

- No changes to `nv_array()` / `nv_scalar()` themselves.
- No dtype-promotion changes — the inferred dtype is whatever
  `nv_array()` / `nv_scalar()` would pick (default `i32` for integers,
  backend-default for doubles, `bool` for logicals).
- No autoconversion of bare vectors (`c(1, 2, 3)`). Users who want this
  must call `nv_array()` explicitly or wrap with `array()`.
- No changes to the `currently_tracing()` branches of the jit impls.
