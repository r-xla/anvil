describe("nv_subset", {
  it("vec[i, drop = <flag>] works", {
    expect_jit_equal(
      nv_tensor(0:4)[1],
      nv_tensor(0L)
    )
    expect_jit_equal(
      nv_tensor(0:4)[1, drop = TRUE],
      nv_scalar(0L)
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
  it("mat[i,j, drop = <flag>] works", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[1, 1],
      nv_tensor(0L, shape = c(1, 1))
    )
    expect_jit_equal(
      x[1, 1, drop = TRUE],
      nv_scalar(0L)
    )
  })
  it("drop does not drop ranges (even 1:1)", {
    expect_jit_equal(
      nv_tensor(0:3)[1:1, drop = TRUE],
      nv_tensor(0L)
    )
  })
  it("mat[tnsr(i), tnsr(j)] works", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[nv_scalar(1L), nv_scalar(1L)],
      nv_tensor(0L, shape = c(1, 1))
    )
    expect_jit_equal(
      x[nv_scalar(1L), nv_scalar(2L)],
      nv_tensor(2L, shape = c(1, 1))
    )
  })
  it("mat[tnsr(i), j] works", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[nv_scalar(1L), 1],
      nv_tensor(0L, shape = c(1, 1))
    )
    expect_jit_equal(
      x[nv_scalar(1L), 2],
      nv_tensor(2L, shape = c(1, 1))
    )
  })

  it("mat[tnsr(i), j:k] works", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[nv_scalar(1L), 1:2],
      nv_tensor(matrix(c(0L, 2L), nrow = 1))
    )
    expect_jit_equal(
      x[nv_scalar(1L), 1:2, drop = TRUE],
      nv_tensor(c(0L, 2L))
    )
  })


  it("mat[i,] works (empty means all)", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[1,],
      nv_tensor(matrix(c(0L, 2L), nrow = 1))
    )
    expect_jit_equal(
      x[2,],
      nv_tensor(matrix(c(1L, 3L), nrow = 1))
    )
  })
  it("mat[i,] with drop works", {
    x <- nv_tensor(matrix(0:3, nrow = 2))
    expect_jit_equal(
      x[1,, drop = TRUE],
      nv_tensor(c(0L, 2L))
    )
    expect_jit_equal(
      x[2,, drop = TRUE],
      nv_tensor(c(1L, 3L))
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
    expect_jit_equal(
      x[1, 1, 1],
      nv_tensor(1L, shape = c(1, 1, 1))
    )
    expect_jit_equal(
      x[1, 1, 1, drop = TRUE],
      nv_scalar(1L)
    )
    expect_jit_equal(
      x[1, 1:2, 1],
      nv_tensor(array(c(1L, 3L), dim = c(1, 2, 1)))
    )
  })
})

describe("nv_subset_assign stablehlo", {
  it("can replace element selected with scalar tensor", {
    expect_jit_equal({
      x <- nv_tensor(1:3)
      x[nv_scalar(1L, dtype = "i64")] <- nv_tensor(-1L)
      x
      },
      nv_tensor(c(-1L, 2L, 3L))
    )
  })
  it("works with i32 indices", {
    expect_jit_equal({
      x <- nv_tensor(1:3)
      x[nv_scalar(1L, dtype = "i32")] <- nv_tensor(-1L)
      x
      },
      nv_tensor(c(-1L, 2L, 3L))
    )
  })
  it("works with literal on rhs", {
    expect_jit_equal({
      x <- nv_tensor(1:2)
      x[1] <- 0L
      x
      },
      nv_tensor(c(0L, 2L))
    )
  })
  it("can assign to contiguous slice in 1D tensor", {
    expect_jit_equal({
      x <- nv_tensor(1:3)
      x[1:2] <- nv_tensor(c(-1L, -2L))
      x
      },
      nv_tensor(c(-1L, -2L, 3L))
    )

  })
  it("can assign to scatter in 1D tensor", {
    expect_jit_equal({
      x <- nv_tensor(1:3)
      x[nv_tensor(c(1L, 3L))] <- nv_tensor(c(-1L, -3L))
      x
      },
      nv_tensor(c(-1L, 2L, -3L))
    )
  })
  it("can scatter with R integer vector", {
    expect_jit_equal({
      x <- nv_tensor(1:3)
      x[c(1, 3)] <- nv_tensor(c(-1L, -3L))
      x
      },
      nv_tensor(c(-1L, 2L, -3L))
    )

  })
  it("checks that update dim is as expected", {
    # maybe we want to allow this, but for now be conservative
    expect_snapshot(error = TRUE,
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
    expect_jit_equal({
      x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
      x[nv_tensor(c(1L, 3L)), 1] <- nv_tensor(c(-1L, -3L))
      x
      },
      nv_tensor(c(-1L, 3L, -3L, 2L, 4L, 6L))
    )
  })
  it("can combine scatter with range", {
    # [[1, 2],
    #  [3, 4],
    #  [5, 6]]
    expect_jit_equal({
      x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
      x[c(1, 3), 1:2] <- nv_tensor(c(-1L, -5L, -2L, -6L), shape = c(2, 2))
      x
      },
      nv_tensor(c(-1, 3, -5, -2, 4, -6), dtype = "i32", shape = c(3, 2))
    )
  })

  it("errs with incorrect subsets", {
    expect_snapshot(error = TRUE,
      jit_eval({
        x <- nv_tensor(matrix(1:6, nrow = 3, byrow = TRUE))
        x[nv_scalar(7L)] <- -99
      })
    )

  })

  it("works correctly with out-of-bound indices", {
    # we just ignore it
    expect_jit_equal({
        x <- nv_tensor(1:6)
        x[nv_scalar(7L)] <- -99L
        x
      },
      nv_tensor(1:6)
    )
  })

  it("broadcasts scalar rhs", {
    expect_jit_equal({
      x <- nv_tensor(1:3)
      x[c(1, 3)] <- -1L
      x
      },
      nv_tensor(c(-1L, 2L, -1L))
    )
  })
  it("Currently supports at most one scattered subset", {
    expect_snapshot(error = TRUE,
      jit_eval({
        x <- nv_tensor(matrix(1:9, nrow = 3, byrow = TRUE))
        x[c(1, 3), c(1, 3)] <- nv_tensor(matrix(c(-1L, -3L, -7L, -9L), nrow = 2, byrow = TRUE))
      })
    )
  })

  it("promotes correctly", {
    # We don't want to convert the lhs, so we only promote rhs to lhs (if possible)
    expect_snapshot(error = TRUE,
      jit_eval({
        x <- nv_tensor(1:3)
        x[1] <- nv_tensor(1)
        x
      })
    )
    expect_jit_equal({
      x <- nv_tensor(1:3)
      x[1] <- nv_tensor(2L, dtype = "i8")
      x
      },
      nv_tensor(c(2L, 2L, 3L))
    )
  })


  it("can replace single element", {
    expect_jit_equal({
      x <- nv_tensor(1:10)
      x[1] <- nv_tensor(-9L)
      x
      },
      nv_tensor(c(-9L, 2:10))
    )
  })
  it("works with literal on rhs", {
    expect_jit_equal({
      x <- nv_tensor(1:2)
      x[1] <- 0L
      x
      },
      nv_tensor(c(0L, 2))
    )

  })
  it("can combine scatter, range and single element in 3D tensor", {
    vals <- array(1:24, dim = c(2, 3, 4))
    vals_exp <- vals
    vals_exp[2, c(1, 3), 3:4] <- -(1:4)

    expect_jit_equal({
      x <- nv_tensor(vals)
      x[2, c(1, 3), 3:4] <- nv_tensor(-(1:4), shape = c(1, 2, 2))
      x
      },
      nv_tensor(vals_exp)
    )
  })

})

describe("nv_subset can be combined with nv_subset_assign", {

})
