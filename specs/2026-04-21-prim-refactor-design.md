# Primitive-layer refactor: `nvl_*`/`p_*` → `prim_*`

**Status:** draft
**Date:** 2026-04-21
**Relates to:** [r-xla/anvl#235](https://github.com/r-xla/anvl/issues/235)

## Motivation

Issue #235 reports confusion between the two layers of the anvl API: the
user-facing `nv_*` functions and the low-level `nvl_*` primitives. The
distinction is real and load-bearing, but the `nv_` / `nvl_` naming gives users
no signal about *what* the layers are for.

This refactor renames the primitive layer to `prim_*` (= primitives), merges
each primitive's metadata object (`p_*`) and traced entry point (`nvl_*`) into
a single callable object, stops exporting the individu  l primitives (users
access them via the exported `prim` environment), and keeps each primitive
visible in the pkgdown reference. We don't export `prim_*` functions to not pollute the namespace too much, but users might still want to access them for implementing their own primitives. All functionality that is covered by the primitives needs to be covered by the `nv_*` API wrappers

## Current state

Each primitive today consists of two separate top-level bindings:

- `p_<name>` — an `AnvlPrimitive` environment holding `name`, `rules`,
  `higher_order`, `subgraphs`. Rules are registered via the
  `[[.AnvlPrimitive` / `[[<-.AnvlPrimitive` methods
  (e.g. `p_add[["stablehlo"]] <- rule`).
- `nvl_<name>` — a jit-compiled function that records the operation into the
  graph via `graph_desc_add(p_<name>, ...)`. Built either by the
  `make_binary_op` / `make_unary_op` / `make_reduce_op` / `make_compare_op`
  helpers or by a direct `jit(function(...) { ... })` call.

Registration is implicit: `.onLoad` scans the namespace for `p_*` bindings and
calls `register_primitive(name, p_<name>)` for each, populating an internal
`prim_dict` environment. Lookup is exposed via the exported function
`prim(name)` (single) or `prim()` (all).

Consumers:

- `R/rules-stablehlo.R`, `R/rules-reverse.R`, `R/rules-quickr.R` mutate each
  `p_*`'s rules via `[[<-`.
- `R/api.R` and `R/api-*.R` call `nvl_<name>(...)` as the lowering step inside
  every user-facing `nv_*` wrapper.
- `R/rules-quickr.R` iterates `lapply(prim(), ...)` to build a quickr-backend
  dispatch table.
- Tests in `tests/testthat/test-primitive*.R` and
  `inst/extra-tests/test-primitives-*-torch.R` exercise `nvl_*` and `prim()`.
- Vignettes `internals.Rmd`, `primitives.Rmd`, `new_primitive.Rmd` teach the
  current API (including `register_primitive()`).
- `pkgdown/_pkgdown.yml` groups primitives via `starts_with("nvl_")` and
  separately lists `register_primitive`.

## Design

### `AnvlPrimitive` (unchanged)

`AnvlPrimitive(name, subgraphs = character())` returns an
environment-backed metadata holder with class `AnvlPrimitive`, carrying
`name`, `rules`, and `subgraphs`. The `[[`, `[[<-`, and `print` S3 methods
keep today's semantics. The redundant `higher_order` field that today's
implementation stores is dropped — derive it from `length(subgraphs) > 0L`
in `is_higher_order_primitive()` instead.

```r
AnvlPrimitive <- function(name, subgraphs = character()) {
  checkmate::assert_string(name)
  checkmate::assert_character(subgraphs)

  env <- new.env(parent = emptyenv())
  env$name <- name
  env$rules <- list()
  env$subgraphs <- subgraphs
  structure(env, class = "AnvlPrimitive")
}

is_higher_order_primitive <- function(x) {
  if (inherits(x, "JitPrimitive")) x <- attr(x, "primitive")
  length(x$subgraphs) > 0L
}
```

This is the object whose rules the `rules-*.R` files populate.

### `new_primitive` helper

`new_primitive()` is the new user-facing constructor. It wires together an
`AnvlPrimitive` metadata object and a jit-wrapped callable, and returns the
combined callable — what `prim_<name>` now is.

```r
new_primitive <- function(name, fn, subgraphs = character(),
                          static = character(), register = TRUE) {
  checkmate::assert_string(name)
  checkmate::assert_function(fn)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  primitive <- AnvlPrimitive(name, subgraphs = subgraphs)
  jit_fn <- jit(fn, static = static, backend = "auto")
  attr(jit_fn, "primitive") <- primitive
  # jit() tags the return with class "JitFunction"; prepend "JitPrimitive"
  # for our S3 dispatch on [[ / [[<- / print.
  class(jit_fn) <- c("JitPrimitive", class(jit_fn))

  if (register) prim[[name]] <- jit_fn
  jit_fn
}
```

Key points:

- `fn` is a plain (non-jit'd) function with the primitive's formals;
  `jit()` is called inside `new_primitive` so `static` is a named arg on
  the helper instead of buried in each call. `backend = "auto"` is
  hardcoded (every current primitive uses `"auto"`).
- No lexical-scope injection. Bodies identify their primitive by passing
  the name string to `graph_desc_add` directly (see the graph-builder
  change below). The duplication of the name once in the `new_primitive`
  call and once inside the body is accepted as the cost of keeping the
  mechanism simple and grep-able.
- Self-registration into `prim` is on by default (`register = TRUE`). Pass
  `register = FALSE` for cases where you want to manage placement manually.

### Graph-builder change

`graph_desc_add()` accepts a primitive-*name* string instead of the
primitive object:

```r
# R/graph.R
graph_desc_add <- function(prim_name, args, params = list(), infer_fn, desc = NULL) {
  primitive <- prim[[prim_name]]     # JitPrimitive callable
  # … everything else the same; PrimitiveCall(primitive, …), print_call_repr,
  # and rule lookup paths all work against the JitPrimitive class.
}

# And the error-display helper updates its label:
print_call_repr <- function(primitive) {
  rlang::exec(call, paste0("prim$", attr(primitive, "primitive")$name))
}
```

Internally, `prim[[name]]` returns the callable; `[[.JitPrimitive` delegates
to the attached `AnvlPrimitive` so rule access
(`primitive[["stablehlo"]]`) continues to work unchanged in lowering code.

### S3 methods on `JitPrimitive`

The callable returned by `new_primitive()` has class
`c("JitPrimitive", "JitFunction")` — `"JitFunction"` is set by `jit()`,
`"JitPrimitive"` prepended in `new_primitive()`. `[[`, `[[<-`, and `print`
delegate to the attached `AnvlPrimitive` so the existing rule-registration
syntax (`prim_add[["stablehlo"]] <- rule`) works unchanged.

```r
`[[.JitPrimitive` <- function(x, name) attr(x, "primitive")[[name]]

`[[<-.JitPrimitive` <- function(x, name, value) {
  attr(x, "primitive")[[name]] <- value
  x                                # same fn, same attached primitive → identity stable
}

print.JitPrimitive <- function(x, ...) {
  print(attr(x, "primitive"))
  invisible(x)
}
```

(`is_higher_order_primitive(x)` is defined above in the `AnvlPrimitive`
section — it unwraps a `JitPrimitive` if needed and derives the answer from
`length(subgraphs) > 0L`.)

### `prim` registry

```r
#' @title Primitive registry
#' @description
#' Environment of all registered primitives. Access via `prim$<name>`
#' (e.g. `prim$add`). Iterate with `as.list(prim)` or `eapply(prim, ...)`.
#' @export
prim <- new.env(parent = emptyenv())
```

`prim` is the single exported public handle for the primitive layer. The old
`prim()` function and `prim_dict` are removed.

### Primitive definitions

For the common binary / unary / reduce / compare patterns, the `make_*`
helpers take the primitive name (a string) and return plain closures that
pass that name to `graph_desc_add`:

```r
make_binary_op <- function(name, stablehlo_infer) {
  force(name); force(stablehlo_infer)
  infer_fn <- function(lhs, rhs) { ... }
  function(lhs, rhs) {
    graph_desc_add(name, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
}

prim_add <- new_primitive("add", make_binary_op("add", stablehlo::infer_types_add))
```

`formals(prim_add)` is `(lhs, rhs)` — the jit wrapper preserves formals.

Custom-body primitives (fill, broadcast_in_dim, dot_general, ..., triangular_solve)
migrate in the same shape. Any `p_<name>` reference inside the body
becomes the literal name string. `static` is lifted to a named argument on
`new_primitive`:

```r
# Before:
p_fill <- AnvlPrimitive("fill")
nvl_fill <- jit(
  function(value, shape, dtype, ambiguous = FALSE, device = NULL) {
    infer_fill <- function(value, shape, dtype, ambiguous) { ... }
    graph_desc_add(p_fill, list(...), infer_fn = infer_fill)[[1L]]
  },
  static = 1:5,
  backend = "auto"
)

# After:
prim_fill <- new_primitive(
  "fill",
  function(value, shape, dtype, ambiguous = FALSE, device = NULL) {
    infer_fill <- function(value, shape, dtype, ambiguous) { ... }
    graph_desc_add("fill", list(...), infer_fn = infer_fill)[[1L]]
  },
  static = 1:5
)
```

Rule-registration files continue to use the top-level `prim_*` binding:

```r
# R/rules-stablehlo.R
prim_add[["stablehlo"]] <- function(inputs, outputs, params, builder) { ... }
```

### Deletions

- `prim()` as a callable function — removed.
- `register_primitive()` — removed from `R/primitive.R` and from `NAMESPACE`.
- `prim_dict` — removed; `prim` replaces it.
- `.onLoad` scan over `p_*` in `R/zzz.R` — removed.

### Exports

- `AnvlPrimitive` — still exported.
- `new_primitive` — newly exported.
- `prim` — newly exported.
- `register_primitive` — unexported (removed).
- Every `prim_<name>` — unexported (drop `@export` from the roxygen block).
  The roxygen block stays so the Rd file is generated.

### Documentation

Replace the existing `section_rules.R` template with a combined
`section_primitive.R` template that produces both the access note and the
implemented-rules list. Every `prim_*` roxygen block uses
`@template section_primitive` (with the existing `@templateVar primitive_id
<name>` line) in place of the old `section_rules` template.

```r
# man-roxygen/section_primitive.R
#' @section Access:
#' Access this primitive via `prim$<%= primitive_id %>`.
#' <% p <- prim[[primitive_id]]; implemented <- Filter(function(r) !is.null(p[[r]]), globals$interpretation_rules) %>
#' <% if (length(implemented) > 0) { %>
#' @section Implemented Rules:
#' <%= paste0("- `", implemented, "`", collapse = "\n#' ") %>
#' <% } %>
```

Uses the exported `prim` env directly (drops the old `getFromNamespace`
indirection). The old `section_rules.R` template is removed.

Examples inside each `prim_*` roxygen block are rewritten to use
`prim$<name>(...)` rather than the old `nvl_<name>(...)` or the now-internal
`prim_<name>(...)` form. Since `prim_<name>` is no longer exported, examples
must hit the exported `prim` env for `R CMD check` to run them.

### pkgdown

`pkgdown/_pkgdown.yml`:

- Replace `starts_with("nvl_")` with `starts_with("prim_")` in the primitives
  section. pkgdown's `starts_with()` matches Rd topic names regardless of
  export status, so the non-exported primitives still appear.
- Remove the `register_primitive` entry.
- Add `prim` to whichever section lists the primitive-layer entry points.

### Tie-in to issue #235

Renaming the layer is the mechanical half of addressing #235; the
documentation half is covered by the vignette rewrite (see "Migration scope"
below), which will explicitly name the two layers (`nv_*` API vs `prim_*`
primitives, accessed via `prim$<name>`) and describe when to reach for which.

## Migration scope

### Code changes

| File | Change |
| --- | --- |
| `R/primitive.R` | Keep `AnvlPrimitive` roughly as-is (env + class + `[[`/`[[<-`/`print`; drop the redundant `higher_order` field). Add `new_primitive()`, `prim` env, `JitPrimitive` S3 class with `[[`/`[[<-`/`print` delegating to the attached `AnvlPrimitive`. Delete `prim()` callable, `register_primitive()`, `prim_dict`. |
| `R/primitives.R` | Migrate ~70 `p_*` / `nvl_*` pairs into single `prim_*` definitions using `new_primitive(name, body, static = ...)`. `make_binary_op`, `make_unary_op`, `make_reduce_op`, `make_compare_op` take a name string and return plain closures (no `jit()`) that pass the name to `graph_desc_add`. |
| `R/zzz.R` | Drop the `p_*` scan in `.onLoad`. |
| `R/graph.R` | `graph_desc_add()` accepts a primitive-*name* string instead of the primitive object, resolves via `prim[[name]]`. `print_call_repr()` updated to show `prim$<name>`. |
| `R/rules-stablehlo.R`, `R/rules-reverse.R`, `R/rules-quickr.R` | `p_<name>[[...]] <- ...` → `prim_<name>[[...]] <- ...` at every rule assignment. `rules-quickr.R`'s `lapply(prim(), ...)` → `eapply(prim, ...)` (or `as.list(prim)`). |
| `R/api.R`, `R/api-*.R` | `nvl_<name>(...)` → `prim_<name>(...)` at every call site. |
| `R/graph-to-quickr.R` | Update any remaining `p_` / `nvl_` references; adjust to the new `graph_desc_add` call shape if it records calls. |
| `NAMESPACE` | Regenerated by `devtools::document()`; loses `nvl_*`, `register_primitive`; gains `prim`. |
| `pkgdown/_pkgdown.yml` | As described above. |

### Tests

- `tests/testthat/test-primitive.R` — rewrite. Current content tests
  `prim("add")`, `register_primitive("add", p, overwrite = TRUE)`,
  `is_higher_order_primitive(prim("while"))`, etc. Replace with
  `prim$add`, `prim$while`, and equivalent coverage for the new
  self-registration path.
- `tests/testthat/test-primitives-stablehlo.R`,
  `test-primitives-reverse.R` — rename `nvl_*` call sites.
- `inst/extra-tests/test-primitives-stablehlo-torch.R`,
  `test-primitives-reverse-torch.R` — rename `nvl_*` call sites.
- Other `tests/testthat/test-*.R` that touch `nvl_*` — rename.

### Vignettes

- `vignettes/internals.Rmd` — update examples `prim("mul")$rules[["reverse"]]`
  → `prim$mul[["reverse"]]`. (Note: `$rules` was a direct env-field access;
  with the new class the supported API is `[[`.) Add an explicit explanation
  of the `nv_*` / `prim_*` layering per issue #235.
- `vignettes/primitives.Rmd` — `prims <- prim()` → `prims <- as.list(prim)`.
- `vignettes/new_primitive.Rmd` — substantive rewrite. The existing vignette
  teaches the two-step `p_repeat_along <- AnvlPrimitive(...)` +
  `register_primitive("repeat_along", p_repeat_along)`. New version is a
  single `prim_repeat_along <- new_primitive("repeat_along", function(...) { ... })`
  call (body references `self`) that covers creation, jit wrapping, and
  registration.

### Scope exclusions

- Graph-machinery change is limited to the `graph_desc_add()` signature
  (primitive-object → primitive-name string) and a one-line tweak to
  `print_call_repr()`. Call recording, rule dispatch, and lowering are
  otherwise untouched.
- No changes to `nv_*` user-facing functions beyond the call-site rename.
- No new primitives or new rules.
- No changes to the stablehlo / pjrt / tengen sibling packages.

## Compatibility

This is a breaking change to the anvl public API:

- `nvl_*` — removed.
- `prim("name")`, `prim()` callable — removed.
- `register_primitive()` — removed.

`prim` is the new way to reach the primitive layer. Users of `nvl_*` in
third-party code will need to rewrite call sites to `prim$<name>`. This is
accepted: anvl is pre-1.0, the layer naming was a documented source of
confusion (#235), and there's no way to deliver the layer-naming fix without
breaking callers.

## Out-of-scope follow-ups

- Writing the #235 documentation contribution to the "Getting Started"
  vignette is partially addressed here (`vignettes/internals.Rmd` update) but
  a dedicated "Two layers" section in `vignettes/anvl.Rmd` is a natural
  next task after this refactor lands.
- Considering whether `make_binary_op` / `make_unary_op` etc. should be
  exposed to extensions that want to define new primitives, or whether the
  `new_primitive(name, body, static = ...)` shape alone is sufficient
  guidance.
