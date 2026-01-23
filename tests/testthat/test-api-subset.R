describe("nv_subset", {
  # Subset specs are passed as ... just like nv_subset:
  #   :          -> full dimension (select all)
  #   scalar int -> single index, drops dimension
  #   start:end  -> contiguous range
  #   list(...)  -> gather indices, preserves dimension
  check_subset <- function(shape, ...) {
    set.seed(1L)
    arr <- array(sample.int(prod(shape) * 10L, prod(shape)), dim = shape)
    quos <- rlang::enquos(...)

    r_args <- vector("list", length(shape))
    drop_dims <- c()
    specs <- list(...)
    for (i in seq_along(shape)) {
      subset <- specs[[i]]
      if (identical(subset, `:`)) {
        r_args[[i]] <- seq(shape[i])
      } else if (is.list(subset)) {
        r_args[[i]] <- unlist(subset)
      } else {
        r_args[[i]] <- subset
        if (is.numeric(subset) && length(subset) == 1L) {
          drop_dims <- c(drop_dims, i)
        }
      }
    }

    r_result <- do.call(`[`, c(list(arr), r_args, list(drop = FALSE)))
    new_dims <- without(dim(r_result), drop_dims)
    if (length(new_dims) == 0L) {
      r_result <- as.vector(r_result)
    } else {
      dim(r_result) <- new_dims
    }

    # Anvil-side: pass quos directly to nv_subset
    x <- nv_tensor(arr)
    anvil_result <- as_array(jit(function(x) {
      rlang::inject(nv_subset(x, !!!quos))
    })(x))
    browser()

    expect_equal(anvil_result, r_result)
  }

  it("1D: single element (drops dim)", {
    check_subset(c(10L), 3L)
  })

  it("1D: range", {
    check_subset(c(10L), 2:5)
  })

  it("1D: multiple non-contiguous indices (gather)", {
    check_subset(c(10L), list(1, 4, 7))
  })

  it("2D: single element in both dims (scalar result)", {
    check_subset(c(4L, 5L), 2L, 3L)
  })

  it("2D: range + full", {
    check_subset(c(6L, 4L), 2:4, `:`)
  })

  it("2D: drop first dim, range in second", {
    check_subset(c(5L, 8L), 3L, 2:6)
  })

  it("2D: gather in first dim, full second dim", {
    check_subset(c(6L, 4L), list(1, 3, 5), `:`)
  })

  it("2D: gather in both dims (cartesian product)", {
    check_subset(c(5L, 6L), list(1, 3, 5), list(2, 4))
  })

  it("2D: gather in one dim, drop the other", {
    check_subset(c(5L, 6L), list(2, 4), 3L)
  })

  it("3D: range, drop, full", {
    check_subset(c(4L, 5L, 3L), 1:3, 2L, `:`)
  })

  it("3D: gather in first two dims, range in third", {
    check_subset(c(4L, 5L, 6L), list(1, 3), list(2, 4, 5), 2:4)
  })

  it("2D: single-element list preserves dim", {
    check_subset(c(4L, 3L), list(2), `:`)
  })

  it("3D: all full (identity)", {
    check_subset(c(3L, 4L, 2L), `:`, `:`, `:`)
  })

  it("4D case with 2 multi-index subset", {
    check_subset(c(3, 4, 2, 4), 1:2, list(1, 2, 4), `:`, list(3, 1))
  })

  it("5D case with 3 multi-index subset and no drop", {
    check_subset(c(3, 4, 2, 4, 3), 1:2, list(1, 2, 4), `:`, list(3, 1), list(2, 2))
  })

  it("5D case with 3 multi-index subset and drop", {
    check_subset(c(3, 4, 2, 4, 3), 1, list(1, 2, 4), `:`, list(3, 1), list(2, 2))
  })

  it("5D case with 3 multi-index subset and 2 drops", {
    check_subset(c(3, 4, 2, 4, 3), 1, list(1, 2, 4), 2, list(3, 1), list(2, 2))
  })

  it("errors on R vector of length > 1", {
    x <- nv_tensor(1:10)
    expect_error(
      jit_eval(x[c(1, 2)]),
      "Vectors of length > 1 are not allowed"
    )
  })
})

describe("nv_subset_assign", {
  check_subset_assign <- function(shape, ...) {
    set.seed(1L)
    arr <- array(sample.int(prod(shape) * 10L, prod(shape)), dim = shape)
    quos <- rlang::enquos(...)
    specs <- list(...)

    # Compute R-side indices and value shape
    r_args <- vector("list", length(shape))
    value_shape <- integer(0L)
    for (i in seq_along(shape)) {
      subset <- if (i <= length(specs)) specs[[i]] else `:`
      if (identical(subset, `:`)) {
        r_args[[i]] <- seq(shape[i])
        value_shape <- c(value_shape, shape[i])
      } else if (is.list(subset)) {
        r_args[[i]] <- unlist(subset)
        value_shape <- c(value_shape, length(subset))
      } else {
        r_args[[i]] <- subset
        value_shape <- c(value_shape, length(subset))
      }
    }

    # Generate random value matching value_shape
    value <- array(sample.int(prod(value_shape) * 10L, prod(value_shape)), dim = value_shape)

    # R-side: assign
    r_result <- do.call(`[<-`, c(list(arr), r_args, list(value = value)))

    # Anvil-side: pass quos directly to nv_subset_assign
    x <- nv_tensor(arr)
    v <- nv_tensor(value)
    anvil_result <- as_array(jit(function(x, v) {
      rlang::inject(nv_subset_assign(x, !!!quos, value = v))
    })(x, v))

    expect_equal(anvil_result, r_result)
  }

  it("1D: single element", {
    check_subset_assign(c(10L), 3L)
  })

  it("1D: range", {
    check_subset_assign(c(10L), 2:5)
  })

  it("1D: full", {
    check_subset_assign(c(6L), `:`)
  })

  it("2D: single element in first dim, full second", {
    check_subset_assign(c(4L, 5L), 2L, `:`)
  })

  it("2D: range in both dims", {
    check_subset_assign(c(6L, 8L), 2:4, 3:6)
  })

  it("2D: single element in both dims", {
    check_subset_assign(c(4L, 5L), 2L, 3L)
  })

  it("2D: full first dim, range in second", {
    check_subset_assign(c(3L, 6L), `:`, 2:4)
  })

  it("3D: range, single, full", {
    check_subset_assign(c(4L, 5L, 3L), 1:3, 2L, `:`)
  })

  it("3D: all full (replace entire tensor)", {
    check_subset_assign(c(2L, 3L, 4L), `:`, `:`, `:`)
  })

  it("2D: trailing dim unspecified (defaults to full)", {
    check_subset_assign(c(4L, 3L), 2:3)
  })

  it("broadcasts scalar rhs", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[1:3] <- -1L
        x
      },
      nv_tensor(c(-1L, -1L, -1L))
    )
  })

  it("1D: gather indices", {
    check_subset_assign(c(5L), list(1, 3, 5))
  })

  it("2D: gather in first dim, full second", {
    check_subset_assign(c(6L, 4L), list(1, 3, 5), `:`)
  })

  it("2D: gather in both dims", {
    check_subset_assign(c(5L, 6L), list(1, 3, 5), list(2, 4))
  })

  it("errors when update shape doesn't match subset shape", {
    expect_error(
      jit_eval({
        x <- nv_tensor(matrix(1:6, nrow = 2))
        x[1:2, 1:2] <- nv_tensor(1:2)
        x
      })
    )
  })
})



describe("subset_specs_start_indices", {
  it("returns all 1s for SubsetFull specs", {
    result <- jit(function() {
      subsets <- list(SubsetFull(5L), SubsetFull(3L))
      subset_specs_start_indices(subsets)
    })()
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(as_array(result)), c(1L, 1L))
  })

  it("returns start values for SubsetRange specs", {
    result <- jit(function() {
      subsets <- list(SubsetRange(3L, 5L), SubsetRange(2L, 4L))
      subset_specs_start_indices(subsets)
    })()
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(as_array(result)), c(3L, 2L))
  })

  it("returns the index for scalar SubsetIndices", {
    result <- jit(function() {
      subsets <- list(SubsetIndex(nv_scalar(4L, dtype = "i64")))
      subset_specs_start_indices(subsets)
    })()
    expect_equal(dtype(result), as_dtype("i64"))
    expect_equal(as.integer(as_array(result)), 4L)
  })

  it("handles mixed spec types", {
    result <- jit(function() {
      subsets <- list(
        SubsetRange(2L, 5L),
        SubsetFull(10L),
        SubsetIndex(nv_scalar(3L, dtype = "i64"))
      )
      subset_specs_start_indices(subsets)
    })()
    expect_equal(dtype(result), as_dtype("i64"))
    expect_equal(as.integer(as_array(result)), c(2L, 1L, 3L))
  })

  it("uses i32 for all-static subsets", {
    result <- jit(function() {
      subsets <- list(SubsetFull(5L), SubsetRange(3L, 5L))
      subset_specs_start_indices(subsets)
    })()
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(as_array(result)), c(1L, 3L))
  })

  it("uses max integer type when dynamic indices are present", {
    result <- jit(function() {
      subsets <- list(
        SubsetIndex(nv_scalar(2L, dtype = "i16")),
        SubsetRange(3L, 5L)
      )
      subset_specs_start_indices(subsets)
    })()
    expect_equal(dtype(result), as_dtype("i32"))
  })

  it("works with a single dimension", {
    result <- jit(function() {
      subsets <- list(SubsetRange(2L, 7L))
      subset_specs_start_indices(subsets)
    })()
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(as_array(result)), 2L)
  })
})
