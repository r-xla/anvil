describe("nv_subset and nv_subset_assign", {
  check <- function(shape, ...) {
    arr <- array(sample.int(prod(shape) * 10L, prod(shape)), dim = shape)

    quos <- rlang::enquos(...)
    spec <- parse_subset_specs(quos, shape)
    r_args <- lapply(seq_along(spec), \(i) {
      drop <- FALSE
      arg <- if (is_subset_full(spec[[i]])) {
        seq(shape[i])
      } else if (is_subset_range(spec[[i]])) {
        (spec[[i]]$start):(spec[[i]]$start + spec[[i]]$size - 1L)
      } else if (is_subset_index(spec[[i]])) {
        drop <- TRUE
        spec[[i]]$index
      } else if (is_subset_indices(spec[[i]])) {
        spec[[i]]$indices
      } else {
        cli_abort("Internal error")
      }
      list(arg, drop)
    })

    args <- lapply(r_args, \(a) a[[1L]])
    drop_dims <- vapply(r_args, \(a) a[[2L]], logical(1L))

    value_shape <- subset_spec_to_shape(spec)

    x <- nv_tensor(arr)

    # test nv_subset
    r_subset_result <- do.call(`[`, c(list(arr), args, list(drop = FALSE)))
    if (!length(value_shape)) {
      r_subset_result <- as.vector(r_subset_result)
    } else {
      dim(r_subset_result) <- value_shape
    }

    static_subset <- as_array(jit(function(x) {
      rlang::inject(nv_subset(x, !!!quos))
    })(x))

    expect_equal(static_subset, r_subset_result)

    # test nv_subset_assign
    n_value <- max(1L, prod(value_shape))
    value <- sample.int(n_value * 10L, n_value)
    if (length(value_shape) > 0L) {
      dim(value) <- value_shape
    }
    r_assign_result <- do.call(`[<-`, c(list(arr), args, list(value = value)))

    v <- nv_tensor(value, shape = value_shape)

    static_assign <- as_array(jit(function(x, v) {
      rlang::inject(nv_subset_assign(x, !!!quos, value = v))
    })(x, v))

    expect_equal(static_assign, r_assign_result)
  }

  it("1D: single element", {
    check(c(10L), 3L)
  })

  it("1D: range", {
    check(c(10L), 2:5)
  })

  it("1D: full", {
    check(c(6L), )
  })

  it("1D: gather", {
    check(c(10L), list(1, 4, 7))
  })

  it("2D: single element in both dims", {
    check(c(4L, 5L), 2L, 3L)
  })

  it("2D: range + full", {
    check(c(6L, 4L), 2:4, )
  })

  it("2D: single in first dim, range in second", {
    check(c(5L, 8L), 3L, 2:6)
  })

  it("2D: single in first dim, full second", {
    check(c(4L, 5L), 2L, )
  })

  it("2D: range in both dims", {
    check(c(6L, 8L), 2:4, 3:6)
  })

  it("2D: full first dim, range in second", {
    check(c(3L, 6L), , 2:4)
  })

  it("???", {
    check(c(2, 3, 2), 1:2, list(1, 3), 1)
  })

  it("2D: gather in first dim, full second", {
    check(c(6L, 4L), list(1, 3, 5), )
  })

  it("2D: gather in both dims", {
    check(c(5L, 6L), list(1, 3, 5), list(2, 4))
  })

  it("2D: gather in one dim, single in the other", {
    check(c(5L, 6L), list(2, 4), 3L)
  })

  it("2D: single-element list preserves dim", {
    check(c(4L, 3L), list(2), )
  })

  it("2D: trailing dim unspecified (defaults to full)", {
    check(c(4L, 3L), 2:3)
  })

  it("3D: range, single, full", {
    check(c(4L, 5L, 3L), 1:3, 2L, )
  })

  it("3D: all full (identity)", {
    check(c(3L, 4L, 2L), , , )
  })

  it("3D: gather in first two dims, range in third", {
    check(c(4L, 5L, 6L), list(1, 3), list(2, 4, 5), 2:4)
  })

  it("4D: 2 multi-index subsets", {
    check(c(3, 4, 2, 4), 1:2, list(1, 2, 4), , list(3, 1))
  })

  it("5D: 3 multi-index subsets, no drop", {
    check(c(3, 4, 2, 4, 3), 1:2, list(1, 2, 4), , list(3, 1), list(2, 2))
  })

  it("5D: 3 multi-index subsets, 1 drop", {
    check(c(3, 4, 2, 4, 3), 1, list(1, 2, 4), , list(3, 1), list(2, 2))
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

  it("errors on out-of-bounds range (start < 1)", {
    x <- nv_tensor(1:10)
    expect_error(jit_eval(x[0:5]), "out of bounds")
  })

  it("errors on out-of-bounds range (end > dim_size)", {
    x <- nv_tensor(1:10)
    expect_error(jit_eval(x[5:11]), "out of bounds")
  })

  it("errors on out-of-bounds single index (< 1)", {
    x <- nv_tensor(1:10)
    expect_error(jit_eval(x[0L]), "out of bounds")
  })

  it("errors on out-of-bounds single index (> dim_size)", {
    x <- nv_tensor(1:10)
    expect_error(jit_eval(x[11L]), "out of bounds")
  })

  it("errors on out-of-bounds list() index", {
    x <- nv_tensor(1:10)
    expect_error(jit_eval(x[list(1, 11)]), "out of bounds")
    expect_error(jit_eval(x[list(0, 5)]), "out of bounds")
  })

  it("works with nv_tensors just like with R indices", {
    x <- jit_eval({
      x <- nv_tensor(1:24, shape = c(2, 3, 4))
      x1 <- x[1:2, list(1, 3), 1]
      x2 <- x[nv_tensor(c(1L, 2L)), list(1L, 3L), nv_scalar(1L)]
      list(x1, x2)
    })
    expect_equal(x[[1]], x[[2]])

    y <- jit_eval({
      x <- nv_tensor(1:24, shape = c(2, 3, 4))
      x1 <- x
      x2 <- x
      update <- nv_tensor(1:4, shape = c(2, 2))
      x1[1:2, list(1, 3), 1] <- update
      x2[nv_tensor(c(1L, 2L)), list(1L, 3L), nv_scalar(1L)] <- update
      list(x1, x2)
    })
    expect_equal(y[[1]], y[[2]])
  })

  it("works with nv_seq", {
    x <- jit_eval({
      x <- nv_tensor(1:10)
      x[nv_seq(2, 5)]
    })
    expect_equal(x, nv_tensor(2:5))
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

  it("works with empty subset", {
    x <- jit_eval({
      nv_tensor(1:10)[]
    })
    expect_equal(x, nv_tensor(1:10))
    x <- jit_eval({
      x <- nv_tensor(1:10)
      x[] <- nv_tensor(2:11)
    })
    expect_equal(x, nv_tensor(2:11))
  })
})
