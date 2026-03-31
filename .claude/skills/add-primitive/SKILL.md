---
name: add-primitive
description: Add a new primitive operation to anvil (nvl_* function with stablehlo, reverse rules, and tests)
user_invocable: true
---

# Add a New Primitive to anvil

You are adding a new primitive operation to the anvil package. Follow each step carefully.

## Step 1: Check StableHLO Support

Before writing any code, verify the corresponding StableHLO operation exists:

1. Check `../stablehlo/R/` for an `op-<name>.R` file (e.g. `op-add.R`).
2. Check that `stablehlo::hlo_<name>()` and `stablehlo::infer_types_<name>()` are exported in `../stablehlo/NAMESPACE`.
3. If the op doesn't exist in stablehlo, stop and tell the user — it must be added there first.
4. Read the StableHLO SPEC (`../stablehlo/SPEC.md`) for the operation's semantics and constraints.

## Step 2: Define the Primitive in `R/primitives.R`

### 2a. Register the primitive

```r
p_<name> <- AnvilPrimitive("<name>")
```

### 2b. Write roxygen2 documentation

Use templates from `man-roxygen/` where applicable:

- **Unary ops:** `@template param_prim_operand_any` (or `_float`, `_signed_numeric`)
- **Binary ops:** `@template params_prim_lhs_rhs_any` (or `_numeric`, `_float`)
- **Return:** `@template return_prim_unary`, `return_prim_binary`, `return_prim_compare`, `return_prim_reduce`
- **Rules section:** `@templateVar primitive_id <name>` + `@template section_rules`
- **StableHLO link:** `@section StableHLO:\n Lowers to [stablehlo::hlo_<name>()].`
- Do NOT mention "1-based indexing" — it's the R default.

### 2c. Implement the function

**Simple unary** (no extra params, e.g. `nvl_abs`, `nvl_negate`):
```r
nvl_<name> <- make_unary_op(p_<name>, stablehlo::infer_types_<name>)
```

**Simple binary** (no extra params, e.g. `nvl_add`, `nvl_mul`):
```r
nvl_<name> <- make_binary_op(p_<name>, stablehlo::infer_types_<name>)
```

**With parameters** (e.g. `nvl_transpose`):
```r
nvl_<name> <- function(operand, <params>) {
  infer_fn <- function(operand, <params>) {
    # Convert 1-based indices to 0-based for stablehlo where needed
    out <- stablehlo::infer_types_<name>(at2vt(operand), ...)[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  graph_desc_add(
    p_<name>,
    list(operand = operand),
    list(<params>),
    infer_fn = infer_fn
  )[[1L]]
}
```

Key rules:
- Propagate `$ambiguous` from inputs to outputs (logical AND for binary ops).
- Use `at2vt()` / `vt2at()` to convert between anvil AbstractArrays and stablehlo ValueTypes.
- Index parameters: convert from 1-based (R) to 0-based (stablehlo) via `- 1L`.

### 2d. Export the function

Add `@export` to the roxygen block. Then run `devtools::document()`.

## Step 3: Implement StableHLO Rule in `R/rules-stablehlo.R`

```r
p_<name>[["stablehlo"]] <- function(<args>) {
  # Convert 1-based indices to 0-based: dims - 1L, permutation - 1L, etc.
  list(stablehlo::hlo_<name>(<args_converted>))
}
```

The rule receives the same arguments as the primitive function. Always convert index parameters from 1-based to 0-based.

## Step 4: Implement Reverse (Differentiation) Rule in `R/rules-backward.R`

```r
p_<name>[["reverse"]] <- function(inputs, outputs, grads, <params>, .required) {
  # inputs:    list of input GraphValues
  # outputs:   list of output GraphValues (the forward pass results)
  # grads:     list of gradient GraphValues (one per output)
  # .required: logical vector — which input gradients are needed
  #
  # Return: list with one element per input (NULL if gradient not needed)
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) <gradient_expression>
  )
}
```

Build gradient expressions using `nvl_*` primitives — never use R arithmetic directly.

For non-differentiable points (e.g. `abs` at 0, `floor` everywhere), follow the conventions used by other frameworks like PyTorch (subgradients, zero gradients, etc.). Read existing reverse rules in `R/rules-backward.R` to see how similar cases are handled.

If the operation is not differentiable at all (e.g. comparisons, boolean ops), still add a reverse rule that returns zeros.

## Step 5: Add an API Wrapper (`nv_*`)

Each primitive should have a corresponding user-facing `nv_*` API function. Follow the `/add-api-function` skill for this step — it covers design principles (R naming, semantics, generics), implementation, documentation, `_pkgdown.yml` placement, and testing.

`nvl_*` primitives are auto-included under the "Primitives" section in `_pkgdown.yml` via `starts_with("nvl_")`.

## Step 6: Write Tests

### Decision: Torch comparison vs. manual R tests

Use this decision framework:

- **Use torch comparison** (`inst/extra-tests/`) when:
  - The operation has a torch equivalent
  - The reverse rule is non-trivial (chain rule, product rule, etc.)
  - Testing numerical correctness across many random inputs matters

- **Use manual R tests** (`tests/testthat/test-primitives-stablehlo.R` and `tests/testthat/test-primitives-reverse.R`) when:
  - The operation is very simple and torch comparison would be overkill
  - No torch equivalent exists
  - The expected output can be stated analytically

Choose one approach, not both.

### Test structure: property-based with edge cases

Tests should be **property-based**: sample diverse inputs (varying shapes, dtypes, edge cases) and verify the anvil result matches torch / R. Use `describe()` / `it()` blocks to organize by scenario.

Edge case sampling is critical — don't just test a single random draw. Structure the test to cover:
- Different shapes (scalar, vector, matrix, 3D)
- Boundary values (zeros, ones, negative, very large/small)
- dtype variations where relevant
- Parameter variations (e.g. different `dims`, `permutation` values)
- Non-differentiable points: when the operation has points where it is not differentiable (e.g. `abs` at 0, `remainder` at integer boundaries), include those values in the test inputs and verify anvil's gradient matches torch's gradient at those points.

### Forward test example (torch comparison in `inst/extra-tests/test-primitives-stablehlo-torch.R`)

```r
describe("p_foo", {
  # Generate inputs that cover important edge cases
  gen_foo <- function(shp, dtype) {
    n <- if (!length(shp)) 1L else prod(shp)
    # Include zeros, negatives, small positives, large values
    vals <- c(0, -1, 1, 0.5, -0.5, 100, -100, sample(rnorm(100), n - 7L))
    vals <- vals[seq_len(n)]
    if (!length(shp)) vals else array(vals, shp)
  }

  it("works for scalars", {
    expect_jit_torch_unary(nvl_foo, torch::torch_foo, integer(), gen = gen_foo)
  })

  it("works for vectors", {
    expect_jit_torch_unary(nvl_foo, torch::torch_foo, 10L, gen = gen_foo)
  })

  it("works for matrices", {
    expect_jit_torch_unary(nvl_foo, torch::torch_foo, c(3, 4), gen = gen_foo)
  })

  it("works for 3D arrays", {
    expect_jit_torch_unary(nvl_foo, torch::torch_foo, c(2, 3, 2), gen = gen_foo)
  })
})
```

For binary ops, use `expect_jit_torch_binary` with `gen_x` / `gen_y`.

### Reverse test example (torch comparison in `inst/extra-tests/test-primitives-reverse-torch.R`)

```r
describe("p_foo", {
  gen_foo <- function(shp, dtype) {
    n <- if (!length(shp)) 1L else prod(shp)
    vals <- c(0.5, -0.5, 1, -1, 2, -2, sample(rnorm(100), max(0, n - 6)))
    vals <- vals[seq_len(n)]
    if (!length(shp)) vals else array(vals, shp)
  }

  it("scalar gradient", {
    verify_grad_uni(nvl_foo, torch::torch_foo, gen = gen_foo)
  })

  it("tensor gradient", {
    verify_grad_uni_tensor(nvl_foo, torch::torch_foo, shape = c(3, 4), gen = gen_foo)
  })

  it("tensor gradient with different shape", {
    verify_grad_uni_tensor(nvl_foo, torch::torch_foo, shape = c(2, 3, 2), gen = gen_foo)
  })
})
```

For binary reverse tests, use `verify_grad_biv` / `verify_grad_biv_tensor` with `gen_lhs` / `gen_rhs`.

### Manual forward test example (in `tests/testthat/test-primitives-stablehlo.R`)

For very simple ops where the expected output is obvious:

```r
describe("p_foo", {
  it("negates the input", {
    x <- nv_array(c(1, -2, 0, 3.5), dtype = "f64")
    out <- as_array(jit(nvl_negate)(x))
    expect_equal(c(out), c(-1, 2, 0, -3.5))
  })

  it("works for scalars", {
    x <- nv_scalar(5, dtype = "f64")
    out <- as_array(jit(nvl_negate)(x))
    expect_equal(c(out), -5)
  })
})
```

### Manual reverse test example (in `tests/testthat/test-primitives-reverse.R`)

```r
describe("p_foo", {
  it("gradient is 1/x", {
    x <- nv_array(c(1, 2, 4), dtype = "f32", shape = 3L)
    f <- function(a) nvl_reduce_sum(nvl_log(a), dims = 1L, drop = TRUE)
    grads <- jit(gradient(f))(x)
    expect_equal(c(as_array(grads[[1L]])), c(1, 0.5, 0.25), tolerance = 1e-6)
  })
})
```

### Key testing helpers

| Helper | File | Purpose |
|--------|------|---------|
| `expect_jit_torch_unary` | `inst/extra-tests/torch-helpers.R` | Compare unary forward with torch |
| `expect_jit_torch_binary` | `inst/extra-tests/torch-helpers.R` | Compare binary forward with torch |
| `verify_grad_uni` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare unary gradient (scalar + tensor) |
| `verify_grad_uni_tensor` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare unary gradient (tensor only) |
| `verify_grad_biv` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare binary gradient (scalar + tensor) |
| `verify_grad_biv_tensor` | `inst/extra-tests/test-primitives-reverse-torch.R` | Compare binary gradient (tensor only) |
| `generate_test_data` | `inst/extra-tests/torch-helpers.R` | Random input sampling by dtype |

Custom generators (`gen`, `gen_x`, `gen_y`, `gen_lhs`, `gen_rhs`) have signature `function(shp, dtype)` and return an R array (or scalar for `integer()` shape).

### Meta-test coverage

The file `tests/testthat/test-primitives-meta.R` automatically checks that every `p_*` primitive has corresponding stablehlo and reverse tests. Your new primitive will be flagged if tests are missing.

## Step 7: Verify

```r
devtools::document()
devtools::load_all()
devtools::test()  # or run specific test files
```

## Checklist

- [ ] StableHLO op exists (`../stablehlo/R/op-<name>.R`)
- [ ] Primitive registered: `p_<name> <- AnvilPrimitive("<name>")`
- [ ] Primitive implemented: `nvl_<name>` with roxygen docs and `@export`
- [ ] StableHLO rule: `p_<name>[["stablehlo"]]` in `R/rules-stablehlo.R`
- [ ] Reverse rule: `p_<name>[["reverse"]]` in `R/rules-backward.R`
- [ ] API wrapper: `nv_<name>` added via `/add-api-function` skill
- [ ] Tests: primitive forward + reverse, property-based with edge cases
- [ ] `devtools::document()` run
- [ ] `devtools::test()` passes
