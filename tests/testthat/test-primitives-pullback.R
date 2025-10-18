test_that("dot_general: vector dot product gradient", {
  # y = <x, y> = sum_i x_i * y_i, scalar output
  f <- function(x, y) {
    nvl_dot_general(
      x,
      y,
      contracting_dims = list(1L, 1L),
      batching_dims = list(integer(), integer())
    )
  }
  g <- jit(gradient(f))
  x <- nv_tensor(c(1, 2, 3), dtype = "f32", shape = 3L)
  y <- nv_tensor(c(4, 5, 6), dtype = "f32", shape = 3L)
  # d/dx = y; d/dy = x
  out <- g(x, y)
  expect_equal(as.numeric(pjrt::as_array(out[[1L]])), c(4, 5, 6))
  expect_equal(as.numeric(pjrt::as_array(out[[2L]])), c(1, 2, 3))
})

test_that("dot_general: matrix-vector with summed loss", {
  nv_hacky_sum <- function(y) {
    ones <- nv_tensor(1, shape = shape(y)[1L], dtype = dtype(y))
    out <- nvl_dot_general(
      y,
      ones,
      contracting_dims = list(1L, 1L),
      batching_dims = list(integer(), integer())
    )
    return(out)
  }

  f <- function(A, x) {
    y <- nvl_dot_general(
      A,
      x,
      contracting_dims = list(2L, 1L),
      batching_dims = list(integer(), integer())
    )
    nv_hacky_sum(y)
  }
  g <- jit(gradient(f))
  A <- nv_tensor(
    matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2),
    dtype = "f32",
    shape = c(2L, 2L)
  )
  x <- nv_tensor(c(5, 6), dtype = "f32", shape = 2L)
  jit(f)(A, x)
  out <- g(A, x)
  dA <- pjrt::as_array(out[[1L]])
  dx <- pjrt::as_array(out[[2L]])

  # dy/dA = [1, 1]^T * x
  dA_true <- outer(c(1, 1), as_array(x))
  # dy/dx = A^T * [1, 1]
  dx_true <- array(t(as_array(A)) %*% c(1, 1), dim = 2L)
  expect_equal(dA, dA_true)
  expect_equal(dx, dx_true)
})

test_that("dot_general: batched matmul gradient w.r.t both inputs", {
  skip_if_not_installed("pjrt")
  # Helpers to reduce repetition
  make_ones_like <- function(Y) {
    nv_tensor(
      rep(1, prod(shape(Y))),
      # TODO: repr() should not be needed
      dtype = dtype(Y),
      shape = shape(Y)
    )
  }
  sum_all <- function(Y) {
    ones <- make_ones_like(Y)
    rank <- length(shape(Y))
    all_dims <- 1:rank
    nvl_dot_general(
      Y,
      ones,
      contracting_dims = list(all_dims, all_dims),
      batching_dims = list(integer(), integer())
    )
  }
  check_case <- function(A, B, contracting_dims, batching_dims) {
    f <- function(A, B) {
      nvl_dot_general(
        A,
        B,
        contracting_dims = contracting_dims,
        batching_dims = batching_dims
      )
    }
    l <- function(A, B) {
      sum_all(f(A, B))
    }
    g <- jit(gradient(l))
    out <- g(A, B)
    dA <- out[[1L]]
    dB <- out[[2L]]

    expect_equal(shape(dA), shape(A))
    expect_equal(shape(dB), shape(B))

    # Verify linearization: <A, dA> == l(A,B) and <B, dB> == l(A,B)
    all_dims_A <- seq_along(shape(A))
    lin_A <- function(A) {
      nvl_dot_general(
        A,
        dA,
        contracting_dims = list(all_dims_A, all_dims_A),
        batching_dims = list(integer(), integer())
      )
    }
    all_dims_B <- seq_along(shape(B))
    lin_B <- function(B) {
      nvl_dot_general(
        B,
        dB,
        contracting_dims = list(all_dims_B, all_dims_B),
        batching_dims = list(integer(), integer())
      )
    }
    expect_equal(jit(lin_A)(A), jit(l)(A, B))
    expect_equal(jit(lin_B)(B), jit(l)(A, B))
  }

  # Case 1: Single contracting dim, no batching dims (existing baseline)
  # A[b,m,k] • B[b,k,n] with contracting k only
  A1 <- nv_tensor(
    1:12,
    shape = c(2, 2, 3),
    dtype = "f32",
  )
  B1 <- nv_tensor(
    1:12,
    shape = c(2, 3, 2),
    dtype = "f32"
  )
  check_case(
    A1,
    B1,
    contracting_dims = list(3L, 2L),
    batching_dims = list(integer(), integer())
  )

  # Case 2: Multiple contracting dims, no batching dims
  # A[b,m,k1,k2] • B[b,k1,k2,n] with contracting (k1,k2)
  A2 <- nv_tensor(
    1:(2 * 2 * 2 * 3),
    shape = c(2, 2, 2, 3),
    dtype = "f32"
  )
  B2 <- nv_tensor(
    1:(2 * 2 * 3 * 2),
    shape = c(2, 3, 2, 2),
    dtype = "f32"
  )
  check_case(
    A2,
    B2,
    contracting_dims = list(c(3L, 4L), c(3L, 2L)),
    batching_dims = list(integer(), integer())
  )

  # Case 3: Multiple batching dims (b1,b2), single contracting dim
  # A[b1,b2,m,k] • B[b1,b2,k,n] with batching (b1,b2) and contracting k
  A3 <- nv_tensor(
    1:(1 * 6 * 3 * 2 * 3),
    shape = c(1, 6, 3, 2, 3),
    dtype = "f32"
  )
  B3 <- nv_tensor(
    1:(6 * 1 * 3 * 2),
    shape = c(6, 1, 3, 2),
    dtype = "f32"
  )
  check_case(
    A3,
    B3,
    contracting_dims = list(5L, 3L),
    batching_dims = list(c(1L, 2L), c(2L, 1L))
  )

  # Case 4: Multiple batching dims and multiple contracting dims
  # A[b1,b2,m,k1,k2] • B[b1,b2,k1,k2,n] with batching (b1,b2) and contracting (k1,k2)
  A4 <- nv_tensor(
    1:(2 * 3 * 2 * 2 * 3),
    shape = c(2, 3, 2, 2, 3),
    dtype = "f32"
  )
  B4 <- nv_tensor(
    1:(2 * 3 * 2 * 3 * 5),
    shape = c(2, 3, 2, 3, 5),
    dtype = "f32"
  )
  check_case(
    A4,
    B4,
    contracting_dims = list(c(4L, 5L), c(3L, 4L)),
    batching_dims = list(c(1L, 2L), c(1L, 2L))
  )

  # batching dims come at last
  A5 <- nv_tensor(
    1:(2 * 3 * 2 * 2 * 3),
    shape = c(2, 3, 2, 2, 3),
    dtype = "f32"
  )
  B5 <- nv_tensor(
    1:(2 * 3 * 2 * 3 * 5),
    shape = c(2, 3, 2, 3, 5),
    dtype = "f32"
  )
  check_case(
    A5,
    B5,
    contracting_dims = list(c(1L, 2L), c(3L, 4L)),
    batching_dims = list(c(4L, 5L), c(1L, 2L))
  )
  # TODO: One super complicated test with many permutations
})

test_that("pullback for comparisons not implemented", {
  g <- gradient(function(a, b) {
    # any comparison; returns boolean but gradient shouldn't be requested
    a < b
  })
  expect_error(g(nv_scalar(1L), nv_scalar(2L)))
})

test_that("atan2 pullback", {
  g <- jit(gradient(nvl_atan2))
  x <- nv_scalar(3.0)
  y <- nv_scalar(4.0)
  out <- g(x, y)
  # d/dx atan2(x, y) = y / (x^2 + y^2)
  # d/dy atan2(x, y) = -x / (x^2 + y^2)
  denom <- 3^2 + 4^2
  expect_equal(out[[1L]], nv_scalar(4 / denom))
  expect_equal(out[[2L]], nv_scalar(-3 / denom))
})
