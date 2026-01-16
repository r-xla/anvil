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
