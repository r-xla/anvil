@../claude-config/CLAUDE.md

## Package Overview

`anvil` is a code transformation framework for R, similar to JAX.
It provides JIT compilation (`jit()`, `xla()`) and automatic differentiation (`gradient()`, `value_and_gradient()`).

## Two-Layer API

- **`nv_*` functions** (e.g. `nv_fill()`, `nv_matmul()`) -- user-facing API in `R/api.R` and `R/api-*.R`. These handle broadcasting, type promotion, default arguments, and then delegate to `nvl_*` primitives.
- **`nvl_*` functions** (e.g. `nvl_fill()`, `nvl_mul()`) -- low-level primitives in `R/primitives.R`. These record operations into the computation graph during tracing.

When adding new functionality, decide which layer it belongs to. Most new operations need both: an `nvl_*` primitive with rules, and an `nv_*` wrapper with R-idiomatic semantics.

## Primitive System

Primitives are `AnvilPrimitive` objects (defined in `R/primitive.R`), stored by convention as `p_<name>` variables. Each primitive has interpretation rules accessed via `p_<name>[["<rule_type>"]]`:

- **`stablehlo`** -- JIT lowering rules in `R/rules-stablehlo.R`. These convert traced operations into StableHLO IR. Since stablehlo uses 0-based indexing, convert indices by subtracting 1.
- **`reverse`** -- Autodiff rules in `R/rules-reverse.R`. Signature: `function(inputs, outputs, grads, .required)`. `grads` contains the upstream gradients (one per output). Return a list of gradients w.r.t. each input, using `NULL` (via `if (.required[[i]])`) for inputs that don't need gradients.
- **`quickr`** -- R-native lowering rules in `R/rules-quickr.R` for the quickr backend.

## Graph Tracing

When a function is JIT-compiled, anvil traces it by executing with `GraphValue` objects instead of real data. Operations record themselves into an `AnvilGraph` (see `R/graph.R`). The graph is then lowered to StableHLO IR or quickr code for compilation.

Key types: `GraphValue` (traced variable), `GraphLiteral` (embedded constant), `AbstractArray` (shape + dtype metadata), `AnvilGraph`.

## NSE and Tracing

When combining non-standard evaluation (NSE) with sub-graph tracing, `force()` all arrayish inputs so they are not accidentally captured as unevaluated promises in the sub-graph descriptor. R's lazy evaluation of function arguments causes hard-to-debug errors otherwise.

## Testing

Each rule of each primitive should be tested. Tests are organized as:

- `tests/testthat/test-primitives-stablehlo.R` -- sources `inst/extra-tests/test-primitives-stablehlo-torch.R`
- `tests/testthat/test-primitives-reverse.R` -- sources `inst/extra-tests/test-primitives-reverse-torch.R`

Prefer testing by comparing with the corresponding torch function. If the test is trivial or the functionality is not covered by torch, test manually instead. Write one or the other, not both.

## Documentation

When writing roxygen2 documentation for primitives or API functions:

- Do not mention "1-based" indexing. Since this is an R package, 1-based indexing is the default.
- Use `@templateVar primitive_id <name>` with `@template section_rules` to auto-generate the "Implemented Rules" section.
- Use `@rdname` or `@inheritParams` to share documentation between `nvl_*` and `nv_*` variants.
- Where a `man-roxygen/` template is too generic for a specific primitive (e.g. the operand has specific dtype constraints), write the `@param` inline instead.
