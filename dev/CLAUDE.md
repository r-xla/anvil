# NA

## Package Overview

`anvil` is a code transformation framework similar to jax for R. It
currently has support for jit compilation and automatic differentiation.

## Development Commands

### Build and Install

``` r
# Load the package for development
devtools::load_all()

# Install the package
devtools::install()

# Build the package (creates tar.gz file)
devtools::build()
```

### Testing

``` r
# Run all tests
devtools::test()

# Run a specific test file
testthat::test_file("tests/testthat/test-constant.R")
```

### Testing Guidelines

Each rule of each primitive should be tested. Many tests can be
implemented by comparing with the corresponding torch function
(inst/extra-tests/test-primitives-stablehlo-torch.R and
inst/extra-tests/test-primitives-backward-torch.R, …). These are sourced
in test-primitives-stablehlo.R and test-primitives-backward.R, etc..
Implement the test by comparing with torch, if possible and necessary.
If the test is very simple, or the functionality not covered by torch,
implement the test manually. Implement either the torch test OR the
manual test, but not both.

### Documentation

``` r
# Generate documentation from roxygen comments
devtools::document()
```

When writing roxygen2 documentation for primitives or API functions:

- Do not mention “1-based” indexing in documentation. Since this is an R
  package, 1-based indexing is the default and stating it is redundant.

- Check the `man-roxygen/` directory for existing templates
  (e.g. `param_operand.R`, `param_shape.R`, `param_dtype.R`,
  `param_ambiguous.R`, `params_lhs_rhs.R`, `section_rules.R`,
  `section_shapes_binary.R`, etc.). Use `@template` to avoid duplicating
  common parameter descriptions.

- Where the template is too generic for a specific primitive (e.g. the
  operand has specific dtype constraints), write the `@param` inline
  instead of using the template.

- Use `@templateVar primitive_id <name>` with `@template section_rules`
  to auto-generate the “Implemented Rules” section.

- Use `@rdname` or `@inheritParams` to inherit documentation from
  related functions where possible, avoiding duplication across `nvl_*`
  and `nv_*` variants.

### Check

``` r
# Run checks for CRAN compliance
devtools::check()
```

## Development Practices

1.  Use S3 (object-oriented system) for defining types and classes.
2.  Follow the established pattern for adding new operations and types
3.  Add tests in `tests/testthat/`
4.  Document functions with roxygen2 comments

## Project Information

1.  `stablehlo` (the jit interpretation rules) uses 0-based indexing,
    but `anvil` uses 1-based indexing. When implementing a jit
    interpretation rule, convert indices by subtracting 1.
2.  The `rules-pullback.R` file contains the differentiation rules for
    the primitive operations. There, `grad` is the gradient of the
    terminal output with respect to the function’s output and the
    function should return the gradient of the terminal output with
    respect to the inputs. The tests are in the file
    `insts/extra-tests/test-primitives-pullback-torch.R`

## Comments

Only add comments if the code is not self-explanatory.

- For length-1 vectors, don’t use
  [`c()`](https://rdrr.io/r/base/c.html). For example, use `1L` instead
  of `c(1L)`.

### Adding new Features

## Adding a Primitive

The functions prefixed with `nvl_` are the primitives and defined in
primitives.R. When implementing a primitive, make sure that the
inference function propagates the ambiguity of the inputs to the output.
Also, check whether the stablehlo package has a corresponding inference
function that can be wrapped. Pay attention that stablehlo uses 0-based
indexing, but `anvil` uses 1-based indexing.

## Adding an API function

API functions are prefixed by `nv_` and are defined in files like api.R
or api-rng.R. Often, they wrap primitives, but make them more convenient
to use. When accessing properties from `tensorish` values, use
`shape_abstract`, `ndims_abstract`, and `dtype_abstract`. Other
accessors are currently not available.

## NSE and Tracing

Whenever we are combining non-standard evaluation (NSE) with tracing of
sub-graphs, we need to [`force()`](https://rdrr.io/r/base/force.html)
the tensorish inputs, so they are not accidentally embedded into the
sub-graphdescriptor. This can happen in R, because the evaluation of
promises in function calls is delayed until they are actually needed,
which causes hard-to-debug errors.

## Pkgdown

When adding a new exported function, ensure it’s in `_pkgdown.yml` file.
