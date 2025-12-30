## Package Overview

`anvil` is a code transformation framework similar to jax for R.
It currently has support for jit compilation and automatic differentiation.

## Development Commands

### Build and Install

```r
# Load the package for development
devtools::load_all()

# Install the package
devtools::install()

# Build the package (creates tar.gz file)
devtools::build()
```

### Testing

```r
# Run all tests
devtools::test()

# Run a specific test file
testthat::test_file("tests/testthat/test-constant.R")
```

### Documentation

```r
# Generate documentation from roxygen comments
devtools::document()
```

### Check

```r
# Run checks for CRAN compliance
devtools::check()
```

## Development Practices

1. Use S7 (object-oriented system) for defining types and classes.
2. Follow the established pattern for adding new operations and types
3. Add tests in `tests/testthat/`
4. Document functions with roxygen2 comments

## Project Information

1. `stablehlo` (the jit interpretation rules) uses 0-based indexing, but `anvil` uses 1-based indexing. When implementing a jit interpretation rule, convert indices by subtracting 1.
2. The `rules-pullback.R` file contains the differentiation rules for the primitive operations.
   There, `grad` is the gradient of the terminal output with respect to the function's output and the function should return the gradient of the terminal output with respect to the inputs.
   The tests are in the file `insts/extra-tests/test-primitives-pullback-torch.R`

## Adding a Primitive

The functions prefixed with `nvl_` are the primitives and defined in primitives.R.
When implementing a primitive, make sure that the inference function propagates the ambiguity of the inputs to the output.
Also, check whether the stablehlo package has a corresponding inference function that can be wrapped.
Pay attention that stablehlo uses 0-based indexing, but `anvil` uses 1-based indexing.
When accessing properties from `tensorish` values, use `shape_abstract`, `ndims_abstract`, and `dtype_abstract`. Other accessors are currently not available.

## Style

* For length-1 vectors, don't use `c()`. For example, use `1L` instead of `c(1L)`.
