describe("nv_subset", {
  it("vec[i] drops dimension (R literal)", {
    # R literal drops dimension
    expect_jit_equal(
      nv_tensor(0:4)[1],
      nv_scalar(0L, ambiguous = FALSE)
    )
  })
  it("vec[list(i)] preserves dimension", {
    # list() preserves dimension
    expect_jit_equal(
      nv_tensor(0:4)[list(1)],
      nv_tensor(0L)
    )
  })
  it("vec[i:j] works", {
    expect_jit_equal(
      nv_tensor(0:4)[1:3],
      nv_tensor(0:2)
    )
    expect_jit_equal(
      nv_tensor(0:4)[2:4],
      nv_tensor(1:3)
    )
  })
  it("mat[i,j] drops both dimensions (R literals)", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[1, 1],
      nv_scalar(0L, ambiguous = FALSE)
    )
  })
  it("mat[list(i), list(j)] preserves both dimensions", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[list(1), list(1)],
      nv_tensor(0L, shape = c(1, 1))
    )
  })
  it("ranges never drop dimensions (even 1:1)", {
    expect_jit_equal(
      nv_tensor(0:3)[1:1],
      nv_tensor(0L)
    )
  })
  it("mat[nv_scalar(i), nv_scalar(j)] drops both dimensions (scalar tensors)", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    # Scalar tensor indices drop dimensions (like R literals)
    expect_jit_equal(
      x[nv_scalar(1L), nv_scalar(1L)],
      nv_scalar(0L, ambiguous = FALSE)
    )
    expect_jit_equal(
      x[nv_scalar(1L), nv_scalar(2L)],
      nv_scalar(2L, ambiguous = FALSE)
    )
  })
  it("mat[nv_scalar(i), j] - both drop (scalar tensor + literal)", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    # Both dims dropped (scalar tensor + literal)
    expect_jit_equal(
      x[nv_scalar(1L), 1],
      nv_scalar(0L, ambiguous = FALSE)
    )
    expect_jit_equal(
      x[nv_scalar(1L), 2],
      nv_scalar(2L, ambiguous = FALSE)
    )
  })

  it("mat[nv_scalar(i), j:k] - scalar tensor drops, range preserves", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    # First dim dropped (scalar tensor), second preserved (range)
    expect_jit_equal(
      x[nv_scalar(1L), 1:2],
      nv_tensor(c(0L, 2L))
    )
  })

  it("mat[nv_tensor(i), j] - 1D tensor preserves, literal drops", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    # 1D tensor indices preserve dimension
    expect_jit_equal(
      x[nv_tensor(1L), 1],
      nv_tensor(0L)
    )
  })

  it("mat[i,] drops first dim, keeps second (R literal + empty)", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[1, ],
      nv_tensor(c(0L, 2L))
    )
    expect_jit_equal(
      x[2, ],
      nv_tensor(c(1L, 3L))
    )
  })
  it("mat[list(i),] preserves first dim", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[list(1), ],
      nv_tensor(matrix(c(0L, 2L), nrow = 1))
    )
    expect_jit_equal(
      x[list(2), ],
      nv_tensor(matrix(c(1L, 3L), nrow = 1))
    )
  })

  it("mat[i:j, k:l] works (multi-dimensional ranges)", {
    x <- nv_tensor(matrix(1:12, nrow = 3, ncol = 4))
    expect_jit_equal(
      x[1:2, 1:2],
      nv_tensor(matrix(c(1L, 2L, 4L, 5L), nrow = 2))
    )
    expect_jit_equal(
      x[2:3, 2:4],
      nv_tensor(matrix(c(5L, 6L, 8L, 9L, 11L, 12L), nrow = 2))
    )
  })

  it("3D tensor slicing works", {
    x <- nv_tensor(array(1:24, dim = c(2, 3, 4)))
    # All R literals drop all dims
    expect_jit_equal(
      x[1, 1, 1],
      nv_scalar(1L, ambiguous = FALSE)
    )
    # All list() preserve all dims
    expect_jit_equal(
      x[list(1), list(1), list(1)],
      nv_tensor(1L, shape = c(1, 1, 1))
    )
    # Mix: literal drops, range preserves
    expect_jit_equal(
      x[1, 1:2, 1],
      nv_tensor(c(1L, 3L))
    )
    # list() preserves
    expect_jit_equal(
      x[list(1), 1:2, list(1)],
      nv_tensor(array(c(1L, 3L), dim = c(1, 2, 1)))
    )
  })

  it("list() with multiple elements gathers non-contiguous indices", {
    x <- nv_tensor(1:10)
    expect_jit_equal(
      x[list(1, 3, 5)],
      nv_tensor(c(1L, 3L, 5L))
    )
  })

  it("2D gather with list() works", {
    x <- nv_tensor(matrix(1:12, nrow = 3, ncol = 4))
    # Gather rows 1 and 3, all columns
    expect_jit_equal(
      x[list(1, 3), ],
      nv_tensor(matrix(c(1L, 3L, 4L, 6L, 7L, 9L, 10L, 12L), nrow = 2))
    )
  })

  it("gather with list() and literal index works", {
    x <- nv_tensor(matrix(1:12, nrow = 3, ncol = 4))
    # Gather rows 1 and 3, column 2 (literal drops column dim)
    expect_jit_equal(
      x[list(1, 3), 2],
      nv_tensor(c(4L, 6L))
    )
  })

  it("errors on R vector of length > 1", {
    x <- nv_tensor(1:10)
    expect_error(
      jit_eval(x[c(1, 2)]),
      "Vectors of length > 1 are not allowed"
    )
  })
})

describe("nv_subset_assign stablehlo", {
  it("can replace element selected with scalar tensor", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[nv_scalar(1L, dtype = "i64")] <- nv_tensor(-1L)
        x
      },
      nv_tensor(c(-1L, 2L, 3L))
    )
  })
  it("works with i32 indices", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[nv_scalar(1L, dtype = "i32")] <- nv_tensor(-1L)
        x
      },
      nv_tensor(c(-1L, 2L, 3L))
    )
  })
  it("works with literal on rhs", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:2)
        x[1] <- 0L
        x
      },
      nv_tensor(c(0L, 2L))
    )
  })
  it("can assign to contiguous slice in 1D tensor", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[1:2] <- nv_tensor(c(-1L, -2L))
        x
      },
      nv_tensor(c(-1L, -2L, 3L))
    )
  })
  it("can assign to scatter in 1D tensor", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[nv_tensor(c(1L, 3L))] <- nv_tensor(c(-1L, -3L))
        x
      },
      nv_tensor(c(-1L, 2L, -3L))
    )
  })
  it("can scatter with R integer vector", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[c(1, 3)] <- nv_tensor(c(-1L, -3L))
        x
      },
      nv_tensor(c(-1L, 2L, -3L))
    )
  })
  it("checks that update dim is as expected", {
    # maybe we want to allow this, but for now be conservative
    expect_error(
      jit_eval({
        x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
        x[nv_tensor(c(1L, 3L)), 1] <- nv_tensor(c(-1L, -3L), shape = 2L)
        x
      })
    )
  })

  it("can combine scatter with index", {
    # [[1, 2],
    #  [3, 4],
    #  [5, 6]]
    jit_eval({
      x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
      x[nv_tensor(c(1L, 3L)), 1] <- nv_tensor(c(-1L, -3L), shape = c(2, 1))
      x
    })

    expect_jit_equal(
      {
        x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
        x[nv_tensor(c(1L, 3L)), 1] <- nv_tensor(c(-1L, -3L), shape = c(2, 1))
        x
      },
      nv_tensor(c(-1L, 3L, -3L, 2L, 4L, 6L), shape = c(3, 2))
    )
  })
  it("can combine scatter with range", {
    # [[1, 2],
    #  [3, 4],
    #  [5, 6]]
    expect_jit_equal(
      {
        x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
        x[c(1, 3), 1:2] <- nv_tensor(c(-1L, -5L, -2L, -6L), shape = c(2, 2))
        x
      },
      nv_tensor(c(-1, 3, -5, -2, 4, -6), dtype = "i32", shape = c(3, 2))
    )
  })

  it("errs with incorrect subsets", {
    expect_error(
      jit_eval({
        x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
        x[nv_scalar(7L)] <- -99
      })
    )
  })

  it("works correctly with out-of-bound indices", {
    # we just ignore it
    expect_jit_equal(
      {
        x <- nv_tensor(1:6)
        x[nv_scalar(7L)] <- -99L
        x
      },
      nv_tensor(1:6)
    )
  })

  it("broadcasts scalar rhs", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[c(1, 3)] <- -1L
        x
      },
      nv_tensor(c(-1L, 2L, -1L))
    )
  })
  it("Currently supports at most one scattered subset", {
    expect_error(
      jit_eval({
        x <- nv_tensor(matrix(1:9, nrow = 3, byrow = TRUE))
        x[c(1, 3), c(1, 3)] <- nv_tensor(matrix(c(-1L, -3L, -7L, -9L), nrow = 2, byrow = TRUE))
      })
    )
  })

  it("promotes correctly", {
    # We don't want to convert the lhs, so we only promote rhs to lhs (if possible)
    expect_snapshot(
      error = TRUE,
      jit_eval({
        x <- nv_tensor(1:3)
        x[1] <- nv_tensor(1)
        x
      })
    )
    expect_jit_equal(
      {
        x <- nv_tensor(1:3)
        x[1] <- nv_tensor(2L, dtype = "i8")
        x
      },
      nv_tensor(c(2L, 2L, 3L))
    )
  })

  it("can replace single element", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:10)
        x[1] <- nv_tensor(-9L)
        x
      },
      nv_tensor(c(-9L, 2:10))
    )
  })
  it("works with literal on rhs", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:2)
        x[1] <- 0L
        x
      },
      nv_tensor(c(0L, 2L))
    )
  })
  it("can combine scatter, range and single element in 3D tensor", {
    vals <- array(1:24, dim = c(2, 3, 4))
    vals_exp <- vals
    vals_exp[2, c(1, 3), 3:4] <- -(1:4)

    expect_jit_equal(
      {
        x <- nv_tensor(vals)
        x[2, c(1, 3), 3:4] <- nv_tensor(-(1:4), shape = c(1, 2, 2))
        x
      },
      nv_tensor(vals_exp)
    )
  })
})

describe("combining subsets with subset-assign", {
  expect_jit_equal(
    {
      x <- nv_tensor(1:4)
      x[1:2][c(2, 1)] <- nv_tensor(c(1L, 2L))
      x
    },
    nv_tensor(c(2L, 1L, 3L, 4L))
  )
})

describe("nv_meshgrid_start_indices", {
  it("works with two scatters", {
    # list(c(1, 3), c(2, 4)) -> combinations: (1,2), (1,4), (3,2), (3,4)
    expect_jit_equal(
      nv_meshgrid_start_indices(list(
        nv_tensor(c(1L, 3L), dtype = "i64"),
        nv_tensor(c(2L, 4L), dtype = "i64")
      )),
      nv_tensor(matrix(c(1L, 2L, 1L, 4L, 3L, 2L, 3L, 4L), nrow = 4, byrow = TRUE), dtype = "i64")
    )
  })

  it("works with scatter and single elements", {
    # list(c(1, 3), 5, 6) -> combinations: (1,5,6), (3,5,6)
    expect_jit_equal(
      nv_meshgrid_start_indices(list(
        nv_tensor(c(1L, 3L), dtype = "i64"),
        nv_scalar(5L, dtype = "i64"),
        nv_scalar(6L, dtype = "i64")
      )),
      nv_tensor(matrix(c(1L, 5L, 6L, 3L, 5L, 6L), nrow = 2, byrow = TRUE), dtype = "i64")
    )
  })

  it("works with multiple scatters and singles", {
    # list(c(1, 3), c(2, 4), 7, 2) -> 4 combinations
    expect_jit_equal(
      nv_meshgrid_start_indices(list(
        nv_tensor(c(1L, 3L), dtype = "i64"),
        nv_tensor(c(2L, 4L), dtype = "i64"),
        nv_scalar(7L, dtype = "i64"),
        nv_scalar(2L, dtype = "i64")
      )),
      nv_tensor(matrix(c(
        1L, 2L, 7L, 2L,
        1L, 4L, 7L, 2L,
        3L, 2L, 7L, 2L,
        3L, 4L, 7L, 2L
      ), nrow = 4, byrow = TRUE), dtype = "i64")
    )
  })

  it("works with all single elements", {
    # list(5, 6, 7) -> single combination: (5,6,7)
    expect_jit_equal(
      nv_meshgrid_start_indices(list(
        nv_scalar(5L, dtype = "i64"),
        nv_scalar(6L, dtype = "i64"),
        nv_scalar(7L, dtype = "i64")
      )),
      nv_tensor(matrix(c(5L, 6L, 7L), nrow = 1), dtype = "i64")
    )
  })

  it("works with three scatters", {
    # 2 x 2 x 2 = 8 combinations
    out <- jit(\() nv_meshgrid_start_indices(list(
      nv_tensor(c(1L, 2L), dtype = "i64"),
      nv_tensor(c(3L, 4L), dtype = "i64"),
      nv_tensor(c(5L, 6L), dtype = "i64")
    )))()
    expect_equal(dim(as_array(out)), c(8, 3))
    arr <- as_array(out)
    # First varies slowest, last varies fastest
    expect_equal(arr[1, ], c(1, 3, 5))
    expect_equal(arr[2, ], c(1, 3, 6))
    expect_equal(arr[3, ], c(1, 4, 5))
    expect_equal(arr[4, ], c(1, 4, 6))
    expect_equal(arr[5, ], c(2, 3, 5))
    expect_equal(arr[6, ], c(2, 3, 6))
    expect_equal(arr[7, ], c(2, 4, 5))
    expect_equal(arr[8, ], c(2, 4, 6))
  })
})

describe("nv_subset with multiple gather dimensions", {
  it("x[list(1, 3), list(2, 4)] gathers all combinations as 2x2 matrix", {
    # 4x4 matrix (column-major in R):
    # [,1] [,2] [,3] [,4]
    # [1,]  1    5    9   13
    # [2,]  2    6   10   14
    # [3,]  3    7   11   15
    # [4,]  4    8   12   16
    x <- nv_tensor(matrix(1:16, nrow = 4))
    # Selecting rows 1,3 and columns 2,4 should give a 2x2 matrix:
    # [,1] [,2]
    # [1,]  5   13   (row 1, cols 2 and 4)
    # [2,]  7   15   (row 3, cols 2 and 4)
    out <- jit(\(x) x[list(1, 3), list(2, 4)])(x)
    expect_equal(dim(as_array(out)), c(2, 2))
    expect_equal(as_array(out), matrix(c(5L, 7L, 13L, 15L), nrow = 2))
  })

  it("x[list(1, 2), list(1, 2)] on 2x2 matrix returns 2x2", {
    # [,1] [,2]
    # [1,]  1    3
    # [2,]  2    4
    x <- nv_tensor(matrix(1:4, nrow = 2))
    out <- jit(\(x) x[list(1, 2), list(1, 2)])(x)
    expect_equal(dim(as_array(out)), c(2, 2))
    # Should be the same as the original matrix
    expect_equal(as_array(out), matrix(1:4, nrow = 2))
  })

  it("x[list(1, 3), list(2, 4), 1] with 3D tensor returns 2x2", {
    x <- nv_tensor(array(1:24, dim = c(3, 4, 2)))
    # Selecting [1,3] x [2,4] x [1] = 2x2 matrix (third dim dropped by literal)
    out <- jit(\(x) x[list(1, 3), list(2, 4), 1])(x)
    expect_equal(dim(as_array(out)), c(2, 2))
  })

  it("x[list(1, 2, 3), list(1, 2)] returns 3x2 matrix", {
    x <- nv_tensor(matrix(1:12, nrow = 4))
    out <- jit(\(x) x[list(1, 2, 3), list(1, 2)])(x)
    expect_equal(dim(as_array(out)), c(3, 2))
  })
})
