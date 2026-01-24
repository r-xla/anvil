describe("nv_subset and nv_subset_assign", {
  check <- function(shape, ...) {
    set.seed(1L)
    arr <- array(sample.int(prod(shape) * 10L, prod(shape)), dim = shape)
    quos <- rlang::enquos(...)
    specs <- list(...)

    # Compute R-side indices, drop_dims, and value_shape
    r_args <- vector("list", length(shape))
    drop_dims <- c()
    value_shape <- integer(0L)
    for (i in seq_along(shape)) {
      subset <- if (i <= length(specs)) specs[[i]] else `:`
      expr <- if (i <= length(quos)) rlang::quo_get_expr(quos[[i]]) else NULL
      is_range_expr <- is.call(expr) && identical(expr[[1]], quote(`:`))
      if (identical(subset, `:`)) {
        r_args[[i]] <- seq(shape[i])
        value_shape <- c(value_shape, shape[i])
      } else if (is.list(subset)) {
        r_args[[i]] <- unlist(subset)
        value_shape <- c(value_shape, length(subset))
      } else {
        r_args[[i]] <- subset
        value_shape <- c(value_shape, length(subset))
        if (is.numeric(subset) && length(subset) == 1L && !is_range_expr) {
          drop_dims <- c(drop_dims, i)
        }
      }
    }

    # Compute dynamic quos (shared by both subset and subset_assign)
    dyn_env <- new.env(parent = environment())
    dyn_quos <- vector("list", length(specs))
    for (i in seq_along(specs)) {
      subset <- specs[[i]]
      if (is.list(subset)) {
        var <- paste0(".dyn_", i)
        dyn_env[[var]] <- nv_tensor(vapply(subset, as.integer, integer(1L)), dtype = "i32")
        dyn_quos[[i]] <- rlang::new_quosure(as.name(var), dyn_env)
      } else if (is.numeric(subset) && length(subset) == 1L) {
        var <- paste0(".dyn_", i)
        dyn_env[[var]] <- nv_scalar(as.integer(subset), dtype = "i32")
        dyn_quos[[i]] <- rlang::new_quosure(as.name(var), dyn_env)
      } else {
        dyn_quos[[i]] <- quos[[i]]
      }
    }

    x <- nv_tensor(arr)

    # --- Test nv_subset ---
    r_subset_result <- do.call(`[`, c(list(arr), r_args, list(drop = FALSE)))
    new_dims <- without(dim(r_subset_result), drop_dims)
    if (length(new_dims) == 0L) {
      r_subset_result <- as.vector(r_subset_result)
    } else {
      dim(r_subset_result) <- new_dims
    }

    static_subset <- as_array(jit(function(x) {
      rlang::inject(nv_subset(x, !!!quos))
    })(x))
    #dynamic_subset <- as_array(jit(function(x) {
    #  rlang::inject(nv_subset(x, !!!dyn_quos))
    #})(x))

    expect_equal(static_subset, r_subset_result)
    #expect_equal(dynamic_subset, r_subset_result)

    # --- Test nv_subset_assign ---
    value <- array(sample.int(prod(value_shape) * 10L, prod(value_shape)), dim = value_shape)
    r_assign_result <- do.call(`[<-`, c(list(arr), r_args, list(value = value)))

    v <- nv_tensor(value)

    static_assign <- as_array(jit(function(x, v) {
      rlang::inject(nv_subset_assign(x, !!!quos, value = v))
    })(x, v))
    dynamic_assign <- as_array(jit(function(x, v) {
      rlang::inject(nv_subset_assign(x, !!!dyn_quos, value = v))
    })(x, v))

    expect_equal(static_assign, r_assign_result)
    expect_equal(dynamic_assign, r_assign_result)
  }

  it("1D: single element", {
    check(c(10L), 3L)
  })

  it("1D: range", {
    check(c(10L), 2:5)
  })

  it("1D: full", {
    check(c(6L), `:`)
  })

  it("1D: gather", {
    check(c(10L), list(1, 4, 7))
  })

  it("2D: single element in both dims", {
    check(c(4L, 5L), 2L, 3L)
  })

  it("2D: range + full", {
    check(c(6L, 4L), 2:4, `:`)
  })

  it("2D: single in first dim, range in second", {
    check(c(5L, 8L), 3L, 2:6)
  })

  it("2D: single in first dim, full second", {
    check(c(4L, 5L), 2L, `:`)
  })

  it("2D: range in both dims", {
    check(c(6L, 8L), 2:4, 3:6)
  })

  it("2D: full first dim, range in second", {
    check(c(3L, 6L), `:`, 2:4)
  })

  it("2D: gather in first dim, full second", {
    check(c(6L, 4L), list(1, 3, 5), `:`)
  })

  it("2D: gather in both dims", {
    check(c(5L, 6L), list(1, 3, 5), list(2, 4))
  })

  it("2D: gather in one dim, single in the other", {
    check(c(5L, 6L), list(2, 4), 3L)
  })

  it("2D: single-element list preserves dim", {
    check(c(4L, 3L), list(2), `:`)
  })

  it("2D: trailing dim unspecified (defaults to full)", {
    check(c(4L, 3L), 2:3)
  })

  it("3D: range, single, full", {
    check(c(4L, 5L, 3L), 1:3, 2L, `:`)
  })

  it("3D: all full (identity)", {
    check(c(3L, 4L, 2L), `:`, `:`, `:`)
  })

  it("3D: gather in first two dims, range in third", {
    check(c(4L, 5L, 6L), list(1, 3), list(2, 4, 5), 2:4)
  })

  it("4D: 2 multi-index subsets", {
    check(c(3, 4, 2, 4), 1:2, list(1, 2, 4), `:`, list(3, 1))
  })

  it("5D: 3 multi-index subsets, no drop", {
    check(c(3, 4, 2, 4, 3), 1:2, list(1, 2, 4), `:`, list(3, 1), list(2, 2))
  })

  it("5D: 3 multi-index subsets, 1 drop", {
    check(c(3, 4, 2, 4, 3), 1, list(1, 2, 4), `:`, list(3, 1), list(2, 2))
  })

  it("5D: 3 multi-index subsets, 2 drops", {
    check(c(3, 4, 2, 4, 3), 1, list(1, 2, 4), 2, list(3, 1), list(2, 2))
  })

  it("1D: first element (boundary)", {
    check(c(8L), 1L)
  })

  it("1D: last element (boundary)", {
    check(c(8L), 8L)
  })

  it("1D: length-1 range preserves dim", {
    check(c(10L), 3:3)
  })

  it("1D: full-length range (equivalent to full)", {
    check(c(6L), 1:6)
  })

  it("1D: gather with non-ascending indices", {
    check(c(10L), list(7, 3, 1))
  })

  it("1D: gather with duplicate indices", {
    check(c(6L), list(2, 2, 4))
  })

  it("2D: dimension of size 1", {
    check(c(1L, 5L), 1L, 2:4)
  })

  it("3D: all dims dropped (scalar result)", {
    check(c(4L, 5L, 3L), 2L, 3L, 1L)
  })

  it("3D: trailing dims unspecified with drop", {
    check(c(4L, 5L, 3L), 2L)
  })

  it("2D: gather with duplicates in both dims", {
    check(c(4L, 5L), list(1, 1, 3), list(2, 2))
  })

  it("2D: boundary indices in both dims", {
    check(c(3L, 4L), 1:3, 1:4)
  })

  it("subset errors on R vector of length > 1", {
    x <- nv_tensor(1:10)
    expect_error(
      jit_eval(x[c(1, 2)]),
      "Vectors of length > 1 are not allowed"
    )
  })

  it("subset_assign broadcasts scalar rhs", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[1:3] <- -1L
        x
      },
      nv_tensor(c(-1L, -1L, -1L))
    )
  })

  it("subset_assign errors when update shape doesn't match", {
    expect_error(
      jit_eval({
        x <- nv_tensor(matrix(1:6, nrow = 2))
        x[1:2, 1:2] <- nv_tensor(1:2)
        x
      }),
      "Update shape does not match subset shape"
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
