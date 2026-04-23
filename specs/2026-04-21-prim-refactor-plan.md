# prim_* Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the anvl primitive layer from `nvl_*` / `p_*` to a single `prim_*` object per primitive, exposed only via an exported `prim` environment (`prim$add`). Keep each primitive visible in pkgdown.

**Architecture:** `AnvlPrimitive(name, subgraphs)` stays the env-based metadata holder (drop the redundant `higher_order` field). A new helper `new_primitive(name, fn, subgraphs, static, register)` builds the `AnvlPrimitive`, wraps `fn` with `jit(backend = "auto")`, attaches the metadata as `attr(., "primitive")`, prepends class `"JitPrimitive"`, and self-registers into the exported `prim` environment. Bodies identify their primitive by passing the name string to `graph_desc_add`, which looks up via `prim[[name]]`. `[[.JitPrimitive` / `[[<-.JitPrimitive` / `print.JitPrimitive` delegate to the attached `AnvlPrimitive`.

**Tech Stack:** R, S3, checkmate, roxygen2, testthat, pkgdown.

**Ref:** `specs/2026-04-21-prim-refactor-design.md` in this repo.

**Strategy:** Introduce the new infrastructure alongside the old, then migrate via temporary back-compat aliases so intermediate states stay green. Only remove the old API once every consumer is on the new one.

---

## Task 1: Baseline check

**Files:** none (verification only).

- [ ] **Step 1: Confirm working tree is clean on `prim-refactor`**

```bash
cd /Users/sebi/r-xla/anvl
git status
```
Expected: `On branch prim-refactor`, "nothing to commit, working tree clean" (the spec commit is the only delta vs main).

- [ ] **Step 2: Run the full R test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass. This is the green baseline the refactor must preserve.

- [ ] **Step 3: Run R CMD check (sanity)**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::check(args = "--no-manual", vignettes = FALSE, error_on = "error")'
```
Expected: 0 errors. Warnings/notes can exist; just note them for comparison at the end.

(No commit.)

---

## Task 2: Add `JitPrimitive` S3 methods

**Files:**
- Modify: `R/primitive.R`
- Test: `tests/testthat/test-primitive.R`

- [ ] **Step 1: Add S3 methods for `JitPrimitive` in `R/primitive.R`**

After the existing `print.AnvlPrimitive` method, append:

```r
#' @method [[ JitPrimitive
#' @export
`[[.JitPrimitive` <- function(x, name) {
  attr(x, "primitive")[[name]]
}

#' @method [[<- JitPrimitive
#' @export
`[[<-.JitPrimitive` <- function(x, name, value) {
  attr(x, "primitive")[[name]] <- value
  x
}

#' @method print JitPrimitive
#' @export
print.JitPrimitive <- function(x, ...) {
  print(attr(x, "primitive"))
  invisible(x)
}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/testthat/test-primitive.R`:

```r
test_that("JitPrimitive [[ delegates to attached AnvlPrimitive", {
  p <- AnvlPrimitive("jp_test_a")
  f <- function(x) x
  attr(f, "primitive") <- p
  class(f) <- c("JitPrimitive", "function")

  # [[<- assigns onto the underlying AnvlPrimitive rules
  f[["stablehlo"]] <- function(x) "stablehlo-rule"
  expect_identical(p[["stablehlo"]](), "stablehlo-rule")

  # [[ reads from the same place
  expect_identical(f[["stablehlo"]], p[["stablehlo"]])
})

test_that("print.JitPrimitive delegates to the AnvlPrimitive", {
  p <- AnvlPrimitive("jp_test_b")
  f <- function(x) x
  attr(f, "primitive") <- p
  class(f) <- c("JitPrimitive", "function")
  expect_output(print(f), "<AnvlPrimitive:jp_test_b>")
})
```

- [ ] **Step 3: Run the tests**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'testthat::test_file("tests/testthat/test-primitive.R")'
```
Expected: all tests pass.

- [ ] **Step 4: Regenerate NAMESPACE**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::document()'
```
Expected: `S3method("[[",JitPrimitive)`, `S3method("[[<-",JitPrimitive)`, `S3method(print,JitPrimitive)` appear in `NAMESPACE`.

- [ ] **Step 5: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/primitive.R tests/testthat/test-primitive.R NAMESPACE
git commit -m "feat: add JitPrimitive S3 methods delegating to AnvlPrimitive

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `new_primitive()` constructor

**Files:**
- Modify: `R/primitive.R`
- Test: `tests/testthat/test-primitive.R`

`new_primitive` registers into the existing internal `prim_dict` env (which the old `prim()` function already reads from). The `prim_dict` → `prim` rename and function deletion happen together in Task 13; keeping them separate now means the existing `prim("name")` consumers continue to work throughout the migration.

- [ ] **Step 1: Write the failing test**

Append to `tests/testthat/test-primitive.R`:

```r
test_that("new_primitive builds a callable that self-registers into prim_dict", {
  on.exit(rm("np_test", envir = prim_dict))

  fn <- new_primitive("np_test", function(x) x + 1)

  expect_class(fn, "JitPrimitive")
  expect_class(fn, "JitFunction")
  expect_identical(prim("np_test"), fn)
  expect_identical(attr(fn, "primitive")$name, "np_test")
  expect_identical(formals(fn), formals(function(x) x + 1))
})

test_that("new_primitive respects register = FALSE", {
  fn <- new_primitive("np_unregistered", function(x) x, register = FALSE)
  expect_false(exists("np_unregistered", envir = prim_dict, inherits = FALSE))
})
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'testthat::test_file("tests/testthat/test-primitive.R")'
```
Expected: the two new tests fail with `could not find function "new_primitive"`.

- [ ] **Step 3: Add `new_primitive()` in `R/primitive.R`**

After the `JitPrimitive` S3 methods (from Task 2), append:

```r
#' @title Create a Primitive
#' @description
#' Builds an [`AnvlPrimitive`] metadata object, wraps `fn` with [`jit()`]
#' (backend `"auto"`), attaches the metadata via `attr(., "primitive")`,
#' prepends class `"JitPrimitive"`, and (by default) registers the result
#' under `name` in the internal primitive registry (exposed via [`prim`]
#' after Task 13).
#' @param name (`character(1)`)\cr Primitive name.
#' @param fn (`function`)\cr Body of the primitive. Its formals become the
#'   formals of the returned JIT-compiled callable. Inside `fn`, identify the
#'   primitive by passing the name string to [`graph_desc_add()`] (not the
#'   `fn` itself).
#' @param subgraphs (`character()`)\cr Names of parameters that are
#'   subgraphs (for higher-order primitives).
#' @param static (`character()` | `integer()`)\cr Passed to [`jit()`].
#' @param register (`logical(1)`)\cr If `TRUE` (default), register the
#'   result under `name` in the primitive registry.
#' @return A callable of class `c("JitPrimitive", "JitFunction")`.
#' @export
new_primitive <- function(name, fn, subgraphs = character(),
                          static = character(), register = TRUE) {
  checkmate::assert_string(name)
  checkmate::assert_function(fn)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  primitive <- AnvlPrimitive(name, subgraphs = subgraphs)
  jit_fn <- jit(fn, static = static, backend = "auto")
  attr(jit_fn, "primitive") <- primitive
  class(jit_fn) <- c("JitPrimitive", class(jit_fn))

  if (register) {
    assign(name, jit_fn, envir = prim_dict)
  }

  jit_fn
}
```

- [ ] **Step 4: Regenerate NAMESPACE + man pages**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::document()'
```
Expected: `export(new_primitive)` in `NAMESPACE`; `man/new_primitive.Rd` created.

- [ ] **Step 5: Re-run the tests**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'testthat::test_file("tests/testthat/test-primitive.R")'
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/primitive.R tests/testthat/test-primitive.R NAMESPACE man/new_primitive.Rd
git commit -m "feat: add new_primitive constructor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Drop `higher_order` field on `AnvlPrimitive`

**Files:**
- Modify: `R/primitive.R`
- Modify: `tests/testthat/_snaps/primitive.md` (snapshot, may auto-update)

- [ ] **Step 1: Simplify `AnvlPrimitive` and rewrite `is_higher_order_primitive`**

In `R/primitive.R`, replace the body of `AnvlPrimitive` (currently setting `env$higher_order` etc.) with:

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
```

And replace `is_higher_order_primitive`:

```r
is_higher_order_primitive <- function(x) {
  if (inherits(x, "JitPrimitive")) x <- attr(x, "primitive")
  length(x$subgraphs) > 0L
}
```

- [ ] **Step 2: Update `subgraphs()` to unwrap `JitPrimitive` if needed**

In the same file, replace `subgraphs <- function(call) { ... }` with:

```r
#' @title Get Subgraphs from Higher-Order Primitive
#' @description
#' Extracts subgraphs from the parameters of a higher-order primitive call.
#' @param call (`PrimitiveCall`)\cr
#'   The primitive call.
#' @return (`list(AnvlGraph)`)\cr
#'   List of subgraphs found in the parameters.
#' @export
subgraphs <- function(call) {
  p <- call$primitive
  if (inherits(p, "JitPrimitive")) p <- attr(p, "primitive")
  if (!is_higher_order_primitive(p)) return(list())

  stats::setNames(
    lapply(p$subgraphs, \(sg) call$params[[sg]]),
    p$subgraphs
  )
}
```

- [ ] **Step 3: Run the existing primitive tests**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'testthat::test_file("tests/testthat/test-primitive.R")'
```
Expected: all pass. If the `expect_snapshot(p)` snapshot differs (because `higher_order` field no longer shown in the env listing), accept with:

```bash
R -q -e 'testthat::snapshot_accept()'
```

- [ ] **Step 4: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/primitive.R tests/testthat/_snaps/primitive.md 2>/dev/null
git commit -m "refactor: derive higher_order from subgraphs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Teach `graph_desc_add()` to accept a name string

**Files:**
- Modify: `R/graph.R`
- Test: `tests/testthat/test-graph.R` (create a small test if none exists for this)

- [ ] **Step 1: Patch `graph_desc_add`**

In `R/graph.R`, replace the existing signature and first lines:

```r
graph_desc_add <- function(prim, args, params = list(), infer_fn, desc = NULL) {
  desc <- desc %||% .current_descriptor(silent = TRUE)
```

with:

```r
graph_desc_add <- function(prim, args, params = list(), infer_fn, desc = NULL) {
  desc <- desc %||% .current_descriptor(silent = TRUE)

  if (is.character(prim)) {
    prim <- get(prim, envir = prim_dict, inherits = FALSE)
  }
  if (inherits(prim, "JitPrimitive")) {
    prim <- attr(prim, "primitive")
  }
```

(Uses `prim_dict` because at this point the old `prim()` callable and the new `prim` env do not yet coexist — `prim_dict` is the canonical registry. Task 14 rewrites this block to use the exported `prim` env.)

The `JitPrimitive` unwrap covers two cases: the name-string lookup returns a `JitPrimitive` callable, and (transiently, during Task 7) callers who pass `p_<name>` — now aliased to the `JitPrimitive` — also land here. Both need to arrive at the `AnvlPrimitive` env for `PrimitiveCall` and `print_call_repr` to see the expected `$name`, `$rules`, etc.

No other changes in this function.

- [ ] **Step 2: Update `print_call_repr` to handle both env and callable**

Replace:

```r
print_call_repr <- function(prim) {
  rlang::exec(call, paste0("nvl_", prim$name))
}
```

with:

```r
print_call_repr <- function(prim) {
  if (inherits(prim, "JitPrimitive")) prim <- attr(prim, "primitive")
  rlang::exec(call, paste0("prim$", prim$name))
}
```

- [ ] **Step 3: Run the full test suite to verify no regression**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests still pass. `graph_desc_add` still accepts the old `p_*` env objects (no consumer has been migrated yet); the new string path is dormant until primitives are migrated.

- [ ] **Step 4: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/graph.R
git commit -m "feat: graph_desc_add accepts primitive name string

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Update `make_*` helpers to take a name string

**Files:**
- Modify: `R/primitives.R`

These helpers are defined at the top of the file (binary/unary) and inline (reduce/compare).

- [ ] **Step 1: Update `make_binary_op` and `make_unary_op`**

Replace (around lines 6-37):

```r
make_binary_op <- function(prim, stablehlo_infer) {
  force(stablehlo_infer)
  infer_fn <- function(lhs, rhs) {
    both_ambiguous <- lhs$ambiguous && rhs$ambiguous
    out <- stablehlo_infer(at2vt(lhs), at2vt(rhs))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- both_ambiguous
    list(out)
  }
  jit(
    function(lhs, rhs) {
      graph_desc_add(prim, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
    },
    backend = "auto"
  )
}

make_unary_op <- function(prim, stablehlo_infer) {
  force(stablehlo_infer)
  infer_fn <- function(operand) {
    out <- stablehlo_infer(at2vt(operand))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  jit(
    function(operand) {
      graph_desc_add(prim, list(operand = operand), infer_fn = infer_fn)[[1L]]
    },
    backend = "auto"
  )
}
```

with (note: no `jit()` — `new_primitive` wraps later):

```r
make_binary_op <- function(name, stablehlo_infer) {
  force(name); force(stablehlo_infer)
  infer_fn <- function(lhs, rhs) {
    both_ambiguous <- lhs$ambiguous && rhs$ambiguous
    out <- stablehlo_infer(at2vt(lhs), at2vt(rhs))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- both_ambiguous
    list(out)
  }
  function(lhs, rhs) {
    graph_desc_add(name, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
}

make_unary_op <- function(name, stablehlo_infer) {
  force(name); force(stablehlo_infer)
  infer_fn <- function(operand) {
    out <- stablehlo_infer(at2vt(operand))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  function(operand) {
    graph_desc_add(name, list(operand = operand), infer_fn = infer_fn)[[1L]]
  }
}
```

- [ ] **Step 2: Update `make_reduce_op` and `make_compare_op`**

Replace `make_reduce_op` (around line 663):

```r
make_reduce_op <- function(name, infer_fn = infer_reduce) {
  force(name); force(infer_fn)
  function(operand, dims, drop = TRUE) {
    graph_desc_add(
      name,
      list(operand = operand),
      params = list(dims = dims, drop = drop),
      infer_fn = infer_fn
    )[[1L]]
  }
}
```

And `make_compare_op` (around line 833):

```r
make_compare_op <- function(name, direction) {
  force(name); force(direction)
  infer_fn <- function(lhs, rhs) infer_compare(lhs, rhs, direction)
  function(lhs, rhs) {
    graph_desc_add(name, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
}
```

- [ ] **Step 3: Do not run tests yet**

The existing `nvl_<name> <- make_binary_op(p_<name>, ...)` lines still pass `p_*` envs as the first argument and the file won't load. This is expected — we fix those call sites together with Task 7.

Commit this step with the next one as a single Task 6+7 commit. Proceed to Task 7 without committing now.

---

## Task 7: Migrate all primitive definitions in `R/primitives.R`

**Files:**
- Modify: `R/primitives.R`

This is the mechanical heart of the refactor. For every current `p_<name>` / `nvl_<name>` pair, replace with a single `prim_<name> <- new_primitive(...)` call. During the migration, leave back-compat aliases for consumers that have not yet been migrated:

```r
# temp aliases — removed in Task 13
p_<name> <- prim_<name>
nvl_<name> <- prim_<name>
```

Rely on `[[.JitPrimitive` delegation so `p_<name>[["stablehlo"]] <- rule` in `rules-*.R` writes through the alias to the underlying `AnvlPrimitive`.

**Patterns to apply:**

**Pattern A — helper-produced primitives (binary/unary/reduce/compare):**

Before:
```r
p_add <- AnvlPrimitive("add")
#' @title …
#' @export
nvl_add <- make_binary_op(p_add, stablehlo::infer_types_add)
```

After:
```r
#' @title …
#' @export
prim_add <- new_primitive("add", make_binary_op("add", stablehlo::infer_types_add))

p_add <- prim_add
nvl_add <- prim_add
```

Apply to every `make_binary_op`, `make_unary_op`, `make_reduce_op`, `make_compare_op`-based primitive (binary, unary, reduce, compare groups; ~45 primitives).

**Pattern B — ad-hoc `jit(…)` primitives (fill, broadcast_in_dim, dot_general, transpose, reshape, concatenate, static_slice, dynamic_slice, dynamic_update_slice, shift_*, bitcast_convert, is_finite, popcnt, clamp, reverse, iota, pad, round, convert, ifelse, print, rng_bit_generator, gather, cholesky, triangular_solve):**

Before:
```r
p_fill <- AnvlPrimitive("fill")
#' @title …
#' @export
nvl_fill <- jit(
  function(value, shape, dtype, ambiguous = FALSE, device = NULL) {
    infer_fill <- function(value, shape, dtype, ambiguous) { ... }
    graph_desc_add(p_fill, list(...), infer_fn = infer_fill)[[1L]]
  },
  static = 1:5,
  backend = "auto"
)
```

After:
```r
#' @title …
#' @export
prim_fill <- new_primitive(
  "fill",
  function(value, shape, dtype, ambiguous = FALSE, device = NULL) {
    infer_fill <- function(value, shape, dtype, ambiguous) { ... }
    graph_desc_add("fill", list(...), infer_fn = infer_fill)[[1L]]
  },
  static = 1:5
)

p_fill <- prim_fill
nvl_fill <- prim_fill
```

Changes:
- drop the separate `p_<name> <- AnvlPrimitive(...)` line.
- wrap the body in `new_primitive("<name>", <body>, static = ..., [subgraphs = ...])`.
- drop the outer `jit(..., backend = "auto")` — `new_primitive` handles jit and backend.
- replace `graph_desc_add(p_<name>, ...)` → `graph_desc_add("<name>", ...)` inside the body.
- keep the old roxygen block intact **for now** (Task 15 rewrites examples and templates).

**Pattern C — higher-order primitives (`if`, `while`, `scatter`):**

Same as Pattern B, but also pass `subgraphs = c(...)` to `new_primitive()`:

Before:
```r
p_if <- AnvlPrimitive("if", subgraphs = c("true_graph", "false_graph"))
#' @title …
#' @export
nvl_if <- jit(
  function(pred, true, false) {
    ...
    out <- graph_desc_add(p_if, list(pred = pred), params = ..., infer_fn = ...)
    ...
  },
  static = 2:3,
  backend = "auto"
)
```

After:
```r
#' @title …
#' @export
prim_if <- new_primitive(
  "if",
  function(pred, true, false) {
    ...
    out <- graph_desc_add("if", list(pred = pred), params = ..., infer_fn = ...)
    ...
  },
  subgraphs = c("true_graph", "false_graph"),
  static = 2:3
)

p_if <- prim_if
nvl_if <- prim_if
```

Apply to `if`, `while`, `scatter` (scatter has `subgraphs = "update_computation_graph"`).

- [ ] **Step 1: Apply Pattern A to all helper-produced primitives (binary/unary/reduce/compare/shift-as-binary)**

Work through `R/primitives.R` top-to-bottom. Do not skip lines. For every `make_binary_op(p_<name>, …)` / `make_unary_op(p_<name>, …)` / `make_reduce_op(p_<name>, …)` / `make_compare_op(p_<name>, …)` line, apply Pattern A. The list by line marker:
`add`, `mul`, `sub`, `negate`, `div`, `pow`, `max`, `min`, `remainder`, `and`, `not`, `or`, `xor`, `atan2`, `abs`, `sqrt`, `rsqrt`, `log`, `tanh`, `tan`, `sine`, `cosine`, `floor`, `ceil`, `sign`, `exp`, `expm1`, `log1p`, `cbrt`, `logistic`, `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, `reduce_any`, `reduce_all`, `eq`, `ne`, `gt`, `ge`, `lt`, `le`.

Note the primitive-name vs alias-name mismatches that must be preserved:
- `nvl_div` → name is `"divide"` (not `"div"`), so the prim-env name is `divide` while the alias is `p_div`/`nvl_div`. Match the spec: `prim_div <- new_primitive("divide", make_binary_op("divide", stablehlo::infer_types_divide))`. Aliases: `p_div <- prim_div; nvl_div <- prim_div`. **Verify the `$name` after migration equals the old `p_div$name`** by inspecting `attr(prim_div, "primitive")$name == "divide"`.
- Same for `pow` → `"power"`, `max` → `"maximum"`, `min` → `"minimum"`, `eq` → `"equal"`, `ne` → `"not_equal"`, `gt` → `"greater"`, `ge` → `"greater_equal"`, `lt` → `"less"`, `le` → `"less_equal"`.

- [ ] **Step 2: Apply Pattern B to ad-hoc `jit()` primitives**

In definition order: `fill`, `broadcast_in_dim`, `dot_general`, `transpose`, `reshape`, `concatenate`, `static_slice`, `dynamic_slice`, `dynamic_update_slice`, `shift_left`, `shift_right_logical`, `shift_right_arithmetic`, `bitcast_convert`, `is_finite`, `popcnt`, `clamp`, `reverse`, `iota`, `pad`, `round`, `convert`, `ifelse`, `print`, `rng_bit_generator`, `gather`, `cholesky`, `triangular_solve`.

For each: collapse `p_<name> <- AnvlPrimitive(...)` + `nvl_<name> <- jit(..., static = ..., backend = "auto")` into a single `prim_<name> <- new_primitive("<name>", fn, static = ...)`; rewrite `graph_desc_add(p_<name>, ...)` → `graph_desc_add("<name>", ...)`; add the two alias lines below.

- [ ] **Step 3: Apply Pattern C to higher-order primitives**

Primitives: `if`, `while`, `scatter`.

For each, the `new_primitive()` call additionally passes `subgraphs = <original vector>`.

- [ ] **Step 4: Load the package to check for syntax errors**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::load_all()'
```
Expected: no errors. Every `prim_<name>` is now defined. Every `p_<name>` and `nvl_<name>` binding still resolves via the alias.

- [ ] **Step 5: Run the full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass. Rules files still write via `p_<name>[[...]] <- ...` through the alias; api files still call `nvl_<name>(...)` through the alias; user-facing tests that called `nvl_*` still work.

- [ ] **Step 6: Commit (tasks 7 + 8 together)**

```bash
cd /Users/sebi/r-xla/anvl
git add R/primitives.R
git commit -m "refactor: migrate primitive definitions to new_primitive

Each p_* / nvl_* pair is replaced with a single prim_* created via
new_primitive(). Temporary aliases (p_<name> <- prim_<name>; nvl_<name>
<- prim_<name>) keep the existing consumers working; they are removed
in a later commit once consumers are migrated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Migrate `R/rules-stablehlo.R`

**Files:**
- Modify: `R/rules-stablehlo.R`

- [ ] **Step 1: Rename every rule assignment**

Use an editor's global replace, scoped to this file: every occurrence of `p_<name>[[` → `prim_<name>[[`. Exactly one regex pass:

Pattern: `^p_(\w+)\[\[` → `prim_\1[[`

Scope: the whole file. ~72 edits.

- [ ] **Step 2: Run the full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass. Writing via `prim_<name>[[<rule>]] <- …` now goes through `[[<-.JitPrimitive` into the attached `AnvlPrimitive`.

- [ ] **Step 3: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/rules-stablehlo.R
git commit -m "refactor: rename p_* -> prim_* in stablehlo rules

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Migrate `R/rules-reverse.R`

**Files:**
- Modify: `R/rules-reverse.R`

- [ ] **Step 1: Rename LHS of every rule assignment**

Same regex replace as Task 8: `^p_(\w+)\[\[` → `prim_\1[[`.

- [ ] **Step 2: Also rename any `nvl_<name>(...)` call sites inside the rule bodies**

Reverse rules frequently compute cotangents by calling the primitive's forward. Replace every `nvl_<name>(` with `prim_<name>(` using: `\bnvl_(\w+)\(` → `prim_\1(`.

- [ ] **Step 3: Run the full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass (reverse rules still use the aliases if any were missed).

- [ ] **Step 4: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/rules-reverse.R
git commit -m "refactor: rename p_* / nvl_* -> prim_* in reverse rules

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Migrate `R/rules-quickr.R`

**Files:**
- Modify: `R/rules-quickr.R`

- [ ] **Step 1: Rename LHS of every rule assignment**

Same regex: `^p_(\w+)\[\[` → `prim_\1[[`.

- [ ] **Step 2: Unwrap `JitPrimitive` when iterating**

The existing `lapply(prim(), function(primitive) { if (is.null(primitive$rules[["quickr"]])) ... else primitive$name })` passes each registered primitive through a lambda that accesses `$rules` and `$name`. After Task 7, the registry now holds `JitPrimitive` callables, so unwrap once at the top of the lambda:

```r
lapply(prim(), function(primitive) {
  if (inherits(primitive, "JitPrimitive")) primitive <- attr(primitive, "primitive")
  if (is.null(primitive$rules[["quickr"]])) {
    NULL
  } else {
    primitive$name
  }
})
```

Keep the `prim()` call form (still a function) — Task 13 converts it to `as.list(prim)` alongside the registry rename.

The `call$primitive` uses later in the file (around line 1452: `call$primitive[["quickr"]]` and line 1459: `call$primitive$name`) read `call$primitive`. `graph_desc_add` (Task 14) will store the `AnvlPrimitive` env there, so these keep working without change. No edits needed inside `quickr_lower_graph_calls`.

- [ ] **Step 3: Run the full test suite (skip quickr tests if not installed)**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/rules-quickr.R
git commit -m "refactor: rename p_* -> prim_* in quickr rules; unwrap JitPrimitive

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Migrate all `R/api*.R` call sites

**Files:**
- Modify: `R/api.R`, `R/api-generics.R`, `R/api-like.R`, `R/api-rng.R`, `R/api-subset.R`, `R/api-utilities.R`
- Modify: `R/reverse.R` (has a couple of `nvl_*` calls)
- Modify: `R/array.R` (has one)
- Modify: `R/utils.R` (has a few)

- [ ] **Step 1: Replace every `nvl_<name>(` with `prim_<name>(` in api files and remaining internal consumers**

Use a regex replace across the listed files (not `R/primitives.R`):
`\bnvl_(\w+)\(` → `prim_\1(`

Exclude `R/primitives.R` from this pass — its own aliases define `nvl_<name>`, and we remove them in Task 13.

- [ ] **Step 2: Run the full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/api.R R/api-generics.R R/api-like.R R/api-rng.R R/api-subset.R R/api-utilities.R R/reverse.R R/array.R R/utils.R
git commit -m "refactor: rename nvl_* -> prim_* at api call sites

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Migrate tests (non-snapshot)

**Files:**
- Modify: `tests/testthat/test-primitive.R` (rewrite the parts that use old API)
- Modify: `tests/testthat/test-primitives-stablehlo.R`
- Modify: `tests/testthat/test-primitives-reverse.R`
- Modify: `inst/extra-tests/test-primitives-stablehlo-torch.R`
- Modify: `inst/extra-tests/test-primitives-reverse-torch.R`
- Modify: any other `tests/testthat/test-*.R` that greps positive for `\bnvl_`

- [ ] **Step 1: Update `tests/testthat/test-primitive.R` for the new class names**

At this point `prim` is still the function form (env promotion happens in Task 13). Rewrite only the bits that break due to `JitPrimitive` replacing `AnvlPrimitive` envs in the registry. Keep `prim("name")` calls — they still work.

Replace the top `test_that("prim", ...)` block (lines 1-14) with:

```r
test_that("prim lookup", {
  expect_identical(prim("add"), prim_add)
  expect_true(is_higher_order_primitive(prim("while")))
  expect_list(prim(), types = "JitPrimitive")
})

test_that("AnvlPrimitive basics", {
  p <- AnvlPrimitive("abc")
  expect_class(p, "AnvlPrimitive")
  expect_equal(p$name, "abc")
  expect_snapshot(p)
})
```

(We drop the `register_primitive` re-register/overwrite bit — that API is removed in Task 13; the snapshot line is kept.)

Replace the `test_that("quickr rules are exposed through primitives", ...)` block with:

```r
test_that("quickr rules are exposed through primitives", {
  expect_true(is.function(prim("add")[["quickr"]]))
  expect_null(prim("print")[["quickr"]])
})
```

Update the `describe("subgraphs", ...)` block — replace `primitive = p_if` with `primitive = prim_if` and `primitive = p_add` with `primitive = prim_add` (still works through the `[[.JitPrimitive` delegation used by the internal graph code).

(The `"documented primitive ids resolve to registered primitives"` block continues to use `prim(id)` — no change needed.)

- [ ] **Step 2: Find and rename `nvl_*` in remaining tests**

```bash
cd /Users/sebi/r-xla/anvl
find tests inst/extra-tests -type f -name '*.R' -exec grep -l '\bnvl_' {} + | \
  xargs sed -i '' -E 's/\bnvl_([a-z_]+)\(/prim_\1(/g'
```
Expected: ~dozens of rewrites. Snapshot `.md` files are excluded by the `-name '*.R'` filter so they are not accidentally mangled.

- [ ] **Step 3: Run the full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass. Snapshot for `AnvlPrimitive` print may have changed (no `higher_order` field); accept if so:

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'testthat::snapshot_accept()'
```

- [ ] **Step 4: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add tests inst/extra-tests
git commit -m "test: migrate tests to prim_* / prim\$name API

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Remove back-compat aliases, promote `prim_dict` to `prim`

**Files:**
- Modify: `R/primitives.R` (remove every `p_<name> <- prim_<name>` and `nvl_<name> <- prim_<name>` line)
- Modify: `R/primitive.R` (rename `prim_dict` → `prim`, export as env; delete old `prim()` callable and `register_primitive()`)
- Modify: `R/zzz.R` (remove the `p_*` scan)
- Modify: `R/rules-quickr.R` (`prim()` → `as.list(prim)` in the one remaining spot)
- Modify: `NAMESPACE` (regenerated)

- [ ] **Step 1: Remove every alias line in `R/primitives.R`**

Regex to match alias lines (two forms):
- `^p_\w+ <- prim_\w+$`
- `^nvl_\w+ <- prim_\w+$`

Delete ~140 alias lines (two per primitive). An editor-global delete-matching-lines is fine, or:

```bash
cd /Users/sebi/r-xla/anvl
sed -i '' -E '/^(p|nvl)_[a-z_]+ <- prim_[a-z_]+$/d' R/primitives.R
```

- [ ] **Step 2: Rename `prim_dict` → `prim` in `R/primitive.R`**

In `R/primitive.R`, replace:

```r
prim_dict <- new.env(parent = emptyenv())
```

with (add the roxygen too):

```r
#' @title Primitive Registry
#' @description
#' Environment containing all registered primitives. Access an individual
#' primitive by name via `prim$add`, `prim$mul`, etc. Iterate over all
#' primitives with `as.list(prim)` or `eapply(prim, f)`.
#' @format An environment.
#' @export
prim <- new.env(parent = emptyenv())
```

Also update `new_primitive()` (from Task 3): change `assign(name, jit_fn, envir = prim_dict)` → `assign(name, jit_fn, envir = prim)`.

- [ ] **Step 3: Delete the old `prim()` callable and `register_primitive()`**

In the same file, remove:
- the entire `register_primitive <- function(name, primitive, overwrite = FALSE) { ... }` block with its `#' @export` roxygen.
- the entire `prim <- function(name = NULL) { ... }` block with its `#' @export` roxygen. The new env-form `prim` added in Step 2 replaces it.

- [ ] **Step 4: Delete the `p_*` scan in `R/zzz.R`**

Remove (around lines 24-28):

```r
  ns <- asNamespace(pkgname)
  for (name in ls(ns, pattern = "^p_")) {
    primitive <- get(name, envir = ns)
    register_primitive(sub("^p_", "", name), primitive)
  }
```

The surrounding `.onLoad` function stays; just delete those 5 lines.

- [ ] **Step 5: Update the last `prim()` call in `R/rules-quickr.R`**

Find the `lapply(prim(), ...)` call (updated in Task 10 to unwrap JitPrimitive). Replace `prim()` with `as.list(prim)`:

```r
lapply(as.list(prim), function(primitive) {
  if (inherits(primitive, "JitPrimitive")) primitive <- attr(primitive, "primitive")
  ...
})
```

- [ ] **Step 5b: Update test-primitive.R to env-form access**

In `tests/testthat/test-primitive.R`, replace the `prim("…")` / `prim()` calls with env-form access (safe now that `prim` is an env):

- `prim("add")` → `prim$add`
- `prim("while")` → `prim$while`
- `prim("print")` → `prim$print`
- `prim()` (list all) → `as.list(prim)`
- `prim(id)` (in the "documented primitive ids" block, where `id` is a variable) → `prim[[id]]`
- The `is.null(prim(id))` check stays functionally correct with `is.null(prim[[id]])`.

- [ ] **Step 6: Regenerate NAMESPACE and man pages**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::document()'
```
Expected: `export(register_primitive)` gone; `man/prim.Rd` now documents the env.

Delete obsolete man files:
```bash
rm -f /Users/sebi/r-xla/anvl/man/register_primitive.Rd
```

- [ ] **Step 7: Run the full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/primitives.R R/primitive.R R/zzz.R R/rules-quickr.R tests/testthat/test-primitive.R NAMESPACE man/
git commit -m "refactor: promote prim_dict to exported prim env; remove old registry API

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Drop backward-compat branch from `graph_desc_add`

**Files:**
- Modify: `R/graph.R`

Now that no caller passes a raw primitive env or callable to `graph_desc_add`, we can tighten the signature.

- [ ] **Step 1: Rewrite `graph_desc_add` to require a name string, and unwrap for `PrimitiveCall`**

Replace the entire `graph_desc_add` function in `R/graph.R` with this final form. Changes from the Task 5 bridge version:

1. Parameter renamed from `prim` to `prim_name` (avoids shadowing the package-level `prim` env).
2. The string-or-object branch is gone; a string is required.
3. The lookup returns the `JitPrimitive` callable. It is unwrapped to the `AnvlPrimitive` env before being stored in `PrimitiveCall`, because `call$primitive$name`, `call$primitive[["quickr"]]`, and similar accesses in `rules-quickr.R` / lowering code expect the env form.

```r
graph_desc_add <- function(prim_name, args, params = list(), infer_fn, desc = NULL) {
  desc <- desc %||% .current_descriptor(silent = TRUE)
  checkmate::assert_string(prim_name)
  jit_primitive <- get(prim_name, envir = prim, inherits = FALSE)
  primitive <- attr(jit_primitive, "primitive")

  boxes_in <- lapply(args, maybe_box_arrayish)
  gnodes_in <- unname(lapply(boxes_in, \(box) box$gnode))
  avals_in <- lapply(boxes_in, \(box) box$gnode$aval)
  ats_out <- tryCatch(
    {
      rlang::exec(infer_fn, !!!c(avals_in, params))
    },
    error = function(e) {
      e$call <- print_call_repr(primitive)
      e <- stablehlo::to_one_based(e)
      rlang::cnd_signal(e)
    }
  )
  gvals_out <- lapply(ats_out, GraphValue)
  call <- PrimitiveCall(primitive, gnodes_in, params, gvals_out)
  desc$calls <- c(desc$calls, list(call))
  lapply(gvals_out, register_gval, desc = desc)
}
```

- [ ] **Step 2: Simplify `print_call_repr` now that its arg is always an `AnvlPrimitive` env**

Replace the Task 5 version (which handled both forms) with:

```r
print_call_repr <- function(prim) {
  rlang::exec(call, paste0("prim$", prim$name))
}
```

- [ ] **Step 3: Run the full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/graph.R
git commit -m "refactor: require name string in graph_desc_add

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Replace `section_rules` template with `section_primitive`

**Files:**
- Create: `man-roxygen/section_primitive.R`
- Delete: `man-roxygen/section_rules.R`
- Modify: every `R/primitives.R` roxygen block — replace `@template section_rules` with `@template section_primitive`

- [ ] **Step 1: Create the new template**

`man-roxygen/section_primitive.R`:

```r
#' @section Access:
#' Access this primitive via `prim$<%= primitive_id %>`.
#' <% p <- prim[[primitive_id]]; implemented <- Filter(function(r) !is.null(p[[r]]), globals$interpretation_rules) %>
#' <% if (length(implemented) > 0) { %>
#' @section Implemented Rules:
#' <%= paste0("- `", implemented, "`", collapse = "\n#' ") %>
#' <% } %>
```

- [ ] **Step 2: Delete the old template**

```bash
rm /Users/sebi/r-xla/anvl/man-roxygen/section_rules.R
```

- [ ] **Step 3: Replace every `@template section_rules` with `@template section_primitive`**

Scoped to `R/primitives.R`:
```bash
cd /Users/sebi/r-xla/anvl
sed -i '' -E 's|#'\'' @template section_rules|#'\'' @template section_primitive|g' R/primitives.R
```
Expected: ~70 substitutions (one per primitive).

- [ ] **Step 4: Regenerate man pages**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::document()'
```
Expected: every `man/prim_*.Rd` now has an "Access" section. No roxygen warnings.

- [ ] **Step 5: Visual spot-check one Rd**

```bash
grep -A2 "Access" /Users/sebi/r-xla/anvl/man/prim_add.Rd
```
Expected: a stanza like `Access this primitive via \code{prim$add}.`

- [ ] **Step 6: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add man-roxygen R/primitives.R man/
git commit -m "docs: merge section_rules into section_primitive template

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Rewrite `@examples` to use `prim$<name>(...)` and drop `@export`

**Files:**
- Modify: `R/primitives.R` — every `prim_*` roxygen block

- [ ] **Step 1: Replace `nvl_<name>(` with `prim$<name>(` inside roxygen examples**

Scoped to `R/primitives.R`, targeting lines that are roxygen comments
(start with `#'`). Use single-quoted sed to avoid shell-escaping `$`:

```bash
cd /Users/sebi/r-xla/anvl
sed -i '' -E "/^#'/ s/\\bnvl_([a-z_]+)\\(/prim\\\$\\1(/g" R/primitives.R
```

Expected: ~70 `nvl_<name>(` inside examples → `prim$<name>(`. Verify with:

```bash
grep -c "prim\$" R/primitives.R
grep -c "nvl_" R/primitives.R
```
The `prim$` count should jump; `nvl_` count in roxygen lines should drop to zero. (Non-roxygen `nvl_<name>` aliases were removed in Task 13.)

- [ ] **Step 2: Drop `@export` from every `prim_*` roxygen block**

Each primitive's roxygen block ends with `#' @export` on its own line
directly above `prim_<name> <- new_primitive(...)`. Remove every such line
using a single-quoted Perl one-liner (so `$1` stays a Perl backref, not a
shell variable):

```bash
cd /Users/sebi/r-xla/anvl
perl -i -0pe 's/#'\''\s+\@export\n(prim_\w+ <- new_primitive\()/$1/g' R/primitives.R
```

Expected: ~70 `@export` lines deleted; `grep -c "^#' @export" R/primitives.R` should return a small number (only for `AnvlPrimitive`, `subgraphs`, `new_primitive`, `prim`, and the handful of other top-level definitions that stay exported — count them manually to confirm nothing unexpected is left over).

- [ ] **Step 3: Regenerate docs**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::document()'
```
Expected: `NAMESPACE` loses all `export(prim_*)` entries; `export(prim)`, `export(new_primitive)`, `export(AnvlPrimitive)` remain.

- [ ] **Step 4: Run full test suite**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::test()'
```
Expected: all tests pass. Examples inside the Rd files still build (pkgdown/example runner reaches them via the exported `prim` env).

- [ ] **Step 5: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add R/primitives.R NAMESPACE man/
git commit -m "refactor: unexport prim_* and rewrite examples to use prim\$<name>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Update vignettes

**Files:**
- Modify: `vignettes/internals.Rmd`
- Modify: `vignettes/primitives.Rmd`
- Modify: `vignettes/new_primitive.Rmd`

- [ ] **Step 1: Update `vignettes/internals.Rmd`**

In `vignettes/internals.Rmd`, line 102 and surrounding paragraph. Replace:

```rmd
We can access a primitive by it's name via the `prim()` function:
```r
prim("mul")$rules[["reverse"]]
```
```

with:

```rmd
We can access a primitive by its name via the `prim` environment:
```r
prim$mul[["reverse"]]
```
```

And similarly replace the second occurrence (line ~127):

```r
prim$mul[["stablehlo"]]
```

Also add a new paragraph above the existing content that names the two layers explicitly (addresses issue #235):

```rmd
anvl has two layers: the user-facing `nv_*` API (in `nv_array()`,
`nv_matmul()`, …) and the low-level primitives `prim_<name>` (accessed
via `prim$<name>`). `nv_*` functions implement R-idiomatic semantics
(broadcasting, type promotion, default arguments) and delegate to
primitives. Primitives are the atomic operations recorded into the
computation graph; each carries its interpretation rules (stablehlo
lowering, reverse-mode autodiff, quickr lowering).
```

- [ ] **Step 2: Update `vignettes/primitives.Rmd`**

In `vignettes/primitives.Rmd`, line 20. Replace:

```r
prims <- prim()
```

with:

```r
prims <- as.list(prim)
```

- [ ] **Step 3: Rewrite `vignettes/new_primitive.Rmd`**

Find the section around line 220-230 that teaches the two-step `AnvlPrimitive` + `register_primitive` dance. Replace with:

```rmd
To define a new primitive, call `new_primitive()`:

```r
prim_repeat_along <- new_primitive(
  "repeat_along",
  function(x, times, axis) {
    infer_fn <- function(x, times, axis) {
      new_shape <- shape(x)
      new_shape[axis] <- new_shape[axis] * times
      list(AbstractArray(dtype = dtype(x), shape = new_shape,
                         ambiguous = x$ambiguous))
    }
    graph_desc_add("repeat_along",
      list(x = x),
      params = list(times = times, axis = axis),
      infer_fn = infer_fn)[[1L]]
  },
  static = 2:3
)
```

The primitive is now callable directly (`prim_repeat_along(x, times, axis)`)
and also accessible via `prim$repeat_along`. Register rules the usual way:

```r
prim_repeat_along[["stablehlo"]] <- function(x, times, axis, builder) { ... }
prim_repeat_along[["reverse"]] <- function(inputs, outputs, grads, .required) { ... }
```
```

Delete any remaining references to `register_primitive()` in the vignette.

- [ ] **Step 4: Build vignettes to verify**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::build_vignettes()'
```
Expected: no errors. Vignette HTML renders.

- [ ] **Step 5: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add vignettes/
git commit -m "docs(vignettes): update for prim_* refactor and explain two layers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Update pkgdown config

**Files:**
- Modify: `pkgdown/_pkgdown.yml`

- [ ] **Step 1: Open `pkgdown/_pkgdown.yml`**

Find the block around line 339 (the primitives section with `starts_with("nvl_")`) and the `register_primitive` entry at line 292.

- [ ] **Step 2: Swap the primitive matcher and add `prim`**

Replace:
```yaml
    desc: Low-level primitive operations (nvl_* functions)
    contents:
      - starts_with("nvl_")
```

with:
```yaml
    desc: Low-level primitive operations (prim_* functions)
    contents:
      - prim
      - new_primitive
      - starts_with("prim_")
```

- [ ] **Step 3: Remove the `register_primitive` entry**

Delete the `- register_primitive` line (around line 292).

- [ ] **Step 4: Build the pkgdown site (optional quick check)**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'pkgdown::build_reference_index()' 2>&1 | head -40
```
Expected: no errors; reference index lists `prim`, `new_primitive`, and every `prim_<name>` topic.

- [ ] **Step 5: Commit**

```bash
cd /Users/sebi/r-xla/anvl
git add pkgdown/_pkgdown.yml
git commit -m "docs(pkgdown): list prim_* primitives and prim env

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Regenerate everything and run full checks**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::document(); devtools::test()'
```
Expected: all tests pass.

- [ ] **Step 2: R CMD check (no manual)**

```bash
cd /Users/sebi/r-xla/anvl
R -q -e 'devtools::check(args = "--no-manual", vignettes = FALSE, error_on = "error")'
```
Expected: 0 errors. Warnings / notes should be no worse than the baseline captured in Task 1.

- [ ] **Step 3: Ensure no stray `nvl_` or `p_<name>` references survive**

```bash
cd /Users/sebi/r-xla/anvl
grep -rn '\bnvl_' R/ tests/ inst/extra-tests/ vignettes/ | grep -v 'nvl_\\*\\|nvl_[* ]' || echo "OK: no nvl_ references"
grep -rn '\bp_[a-z]' R/ | grep -v '# ' || echo "OK: no p_ references"
grep -rn '\bregister_primitive\b\|\bprim_dict\b' R/ tests/ inst/extra-tests/ vignettes/ || echo "OK: no old-API references"
```
Expected: "OK" for each. Any hits need a follow-up commit.

- [ ] **Step 4: Run `jarl check`**

```bash
cd /Users/sebi/r-xla/anvl
jarl check . || true
```
Review output; if new lint errors appeared that weren't present at baseline, fix them in a follow-up commit.

- [ ] **Step 5: Summary commit if anything got picked up**

If any of Steps 3-4 surfaced a follow-up:

```bash
cd /Users/sebi/r-xla/anvl
git add -u
git commit -m "fixup: straggling references after prim_* refactor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Push and open PR**

```bash
cd /Users/sebi/r-xla/anvl
git push -u origin prim-refactor
gh pr create --title "refactor: rename nvl_* / p_* to prim_*" --body "$(cat <<'EOF'
## Summary
- Merges the primitive metadata (`p_*`) and traced entry point (`nvl_*`) into a single `prim_*` object per primitive.
- Unexports individual primitives; users access via `prim$<name>` on the newly exported `prim` environment.
- Introduces `new_primitive()` as the single constructor for primitives; `AnvlPrimitive()` keeps its role as the metadata holder.
- `graph_desc_add()` now identifies its primitive by name string.
- Merges the rules/access doc sections into a single `section_primitive` roxygen template.
- Closes #235 (the layer naming is now self-describing; `internals.Rmd` documents the two layers explicitly).

## Test plan
- [x] `devtools::test()` — all tests pass.
- [x] `devtools::check()` — 0 errors.
- [x] Vignettes build (`devtools::build_vignettes()`).
- [x] pkgdown reference lists every `prim_*` topic plus `prim` and `new_primitive`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Appendix: Primitive name↔alias mismatches (for Task 7)

These primitives have an internal name that differs from the R binding:

| Binding | Internal name |
| --- | --- |
| `prim_div` | `divide` |
| `prim_pow` | `power` |
| `prim_max` | `maximum` |
| `prim_min` | `minimum` |
| `prim_eq`  | `equal` |
| `prim_ne`  | `not_equal` |
| `prim_gt`  | `greater` |
| `prim_ge`  | `greater_equal` |
| `prim_lt`  | `less` |
| `prim_le`  | `less_equal` |

For each of these, `new_primitive()` takes the **internal** name (first column of the original `AnvlPrimitive(...)` call), and `graph_desc_add(...)` inside the body passes the same internal name string. The R binding (`prim_div`, etc.) is what we write on the left-hand side of the assignment.

After migration verify:
```r
attr(prim_div, "primitive")$name == "divide"   # TRUE
prim[["divide"]] == prim_div                   # TRUE (identity)
```
