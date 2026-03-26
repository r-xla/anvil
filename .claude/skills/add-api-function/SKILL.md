---
name: add-api-function
description: Add a user-facing nv_* API function to anvil, wrapping primitives with R-idiomatic semantics
user_invocable: true
---

# Add an API Function (`nv_*`) to anvil

You are adding a user-facing API function to the anvil package. API functions (`nv_*`) wrap one or more primitives (`nvl_*`) to provide a convenient, R-idiomatic interface.

## Design Principles

### Follow R semantics

- **Naming:** Use R naming conventions. If base R or a common R package already has a function for this operation, match its name. For example: `nv_abs` (not `nv_absolute`), `nv_transpose` (matching `t()`), `nv_seq` (matching `seq()`). Only deviate from R names when there is a good reason (e.g. no R equivalent, or the R name would be ambiguous in the array context).
- **Semantics:** Match R behavior where it makes sense. For example, `nv_seq(start, end)` mirrors R's `seq()`, reductions like `nv_reduce_sum()` map to `sum()`. When R semantics conflict with array programming conventions (e.g. recycling rules vs. explicit broadcasting), prefer the array convention but document the difference.
- **R generics:** If a base R generic exists for this operation, implement an S3 method. For example:
  - `t()` → `t.AnvilBox` / `t.AnvilArray` dispatching to `nv_transpose()`
  - `abs()` → handled via `Math.AnvilBox` group generic
  - `+`, `-`, `*`, `/` → handled via `Ops.AnvilBox` group generic
  - `sum()`, `prod()`, `min()`, `max()` → handled via `Summary.AnvilBox` group generic
  - `[` → `[.AnvilBox` dispatching to `nv_subset()`

  Check `R/api-generics.R` for the existing group generics (`Ops`, `Math`, `Summary`) and individual method registrations. If your operation fits an existing group generic, add it there. Otherwise, create a standalone S3 method.

### Propose, then confirm

The exact convenience a wrapper should add varies by operation. **Propose a wrapper to the user but ask them to confirm** the semantic differences before implementing. Common patterns include:

- **Type promotion:** auto-promote inputs to a common dtype via `nv_promote_to_common()`
- **Broadcasting:** broadcast scalars to match array shapes via `nv_broadcast_scalars()`
- **Default arguments:** infer `dtype` from the input when not provided
- **Idempotency:** skip no-op cases (return operand unchanged if already correct dtype/shape)
- **Input coercion:** convert auxiliary arguments to match the operand's dtype

## Implementation

### Where to put the code

- Most API functions go in `R/api.R`.
- RNG functions go in `R/api-rng.R`.
- Subsetting goes in `R/api-subset.R`.
- S3 method registrations go in `R/api-generics.R`.

For simple binary ops, use the factory:
```r
nv_<name> <- make_do_binary(nvl_<name>)
```
This automatically adds type promotion and scalar broadcasting.

For simple unary ops that need no extra convenience, alias the primitive directly:
```r
nv_<name> <- nvl_<name>
```

For ops needing custom logic, write a function. Use `shape_abstract()`, `ndims_abstract()`, and `dtype_abstract()` to access properties from arrayish values.

## Roxygen2 Documentation

API functions use a consistent documentation pattern. Use templates from `man-roxygen/` where applicable.

### Structure

```r
#' @title <Short Title>
#' @description
#' <One-sentence description.> You can also use `<R operator or generic>()`.
#' @template param_operand              # or @template params_lhs_rhs, etc.
#' @param <custom_param> (<type>)\cr    # for params not covered by templates
#'   <Description.>
#' @template return_unary               # or return_binary, return_reduce, etc.
#' @seealso [nvl_<name>()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   <example code>
#' })
#' @export
```

### Key conventions

- **`@title`**: short, e.g. "Absolute Value", "Addition", "Transpose"
- **`@description`**: one sentence describing what the function does. If an R operator or generic dispatches to this function, mention it: "You can also use `abs()`.", "You can also use the `+` operator."
- **`@template`**: use templates for common parameter/return patterns:
  - `param_operand` — single operand
  - `params_lhs_rhs` — binary operands (includes promotion/broadcasting note)
  - `param_dtype`, `param_shape`, `param_ambiguous` — common params
  - `return_unary`, `return_binary`, `return_reduce`, `return_reduce_boolean`
  - `params_reduce` — dims + drop params for reductions
- **`@param`**: write inline for parameters not covered by templates
- **`@seealso`**: always link to the underlying `nvl_*` primitive. Optionally link to related `nv_*` functions.
- **`@examplesIf pjrt::plugin_is_downloaded()`**: wrap examples in this guard. Use `jit_eval({...})` for concise examples.
- **`@family`**: use for groups of related functions (e.g. `@family rng` for all RNG functions)

### S3 methods for R generics

When implementing an R generic, unify documentation using `@name` and `@rdname`:

```r
#' @rdname nv_<name>
#' @export
<generic>.<class> <- function(x, ...) {
  nv_<name>(x, ...)
}
```

The main documentation lives on the `nv_*` function; S3 methods use `@rdname` to point there.

## Add to `_pkgdown.yml`

The `nv_*` function must be added to the appropriate semantic section in `_pkgdown.yml` (e.g. "Arithmetic operations", "Mathematical functions", "Reduction operations", "Linear algebra", etc.). Check the existing sections and pick the best fit.

## Write Tests (in `tests/testthat/test-api.R`)

Add a **forward-pass-only** test for the `nv_*` wrapper. Focus on what makes the wrapper different from the primitive — the convenience behavior it adds. No need to test gradients (those are covered by the primitive tests).

Use `describe()` / `it()` blocks. Test the specific convenience features:

```r
describe("nv_foo", {
  it("promotes dtypes automatically", {
    out <- jit(nv_foo)(nv_array(1L, dtype = "i32"), nv_array(1.5, dtype = "f32"))
    expect_equal(dtype(out), "f32")
  })

  it("broadcasts scalar to array", {
    out <- jit(nv_foo)(nv_scalar(2), nv_array(c(1, 2, 3)))
    expect_equal(shape(out), 3L)
  })

  it("works via the + operator", {
    out <- jit_eval({
      nv_array(c(1, 2)) + nv_array(c(3, 4))
    })
    expect_equal(as_array(out), array(c(4, 6), dim = 2L))
  })
})
```

## Verify

```r
devtools::document()
devtools::load_all()
devtools::test()
```

## Checklist

- [ ] Design confirmed with user (naming, semantics, convenience features)
- [ ] R generic / S3 method added if applicable (in `R/api-generics.R`)
- [ ] `nv_<name>` implemented with roxygen docs and `@export`
- [ ] `_pkgdown.yml`: added to appropriate semantic section
- [ ] Forward-pass test in `tests/testthat/test-api.R`
- [ ] `devtools::document()` run
- [ ] `devtools::test()` passes
