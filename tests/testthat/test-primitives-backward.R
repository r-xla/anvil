# most pullback tests are in extra-tests for comparison with torch
# Just a few selected ones that don't use torch are here

test_that("p_dot_general: vector dot product gradient", {
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

test_that("p_dot_general: matrix-vector with summed loss", {
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

test_that("p_dot_general: batched matmul gradient w.r.t both inputs", {
  skip_if_metal()
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
})

test_that("broadcasting", {
  f <- jit(gradient(
    function(x, y) {
      mean(x + y)
    },
    wrt = c("x", "y")
  ))

  out <- f(nv_scalar(1), nv_tensor(0, shape = c(1, 2)))
  expect_equal(out[[1L]], nv_scalar(1))
  expect_equal(out[[2L]], nv_tensor(0.5, shape = c(1, 2)))
})

test_that("p_if", {
  # TODO:
  #f <- jit(gradient(
  #  function(pred, x) {
  #    nvl_if(pred, x * nv_scalar(1), x * nv_scalar(2))
  #  },
  #  wrt = "x"
  #))
  #out <- f(nv_scalar(TRUE), nv_scalar(2))
  #expect_equal(out[[1L]], nv_scalar(2))
  #expect_equal(out[[2L]], nv_scalar(1))
})

test_that("p_log backward", {
  f <- jit(gradient(function(x) {
    nv_log(x)
  }))

  x <- nv_scalar(2, dtype = "f32")
  grad <- f(x)[[1L]]

  expect_equal(as_array(grad), 0.5)
})

test_that("p_exp", {
  f <- jit(gradient(function(x) {
    nv_exp(x)
  }))

  x <- nv_scalar(2, dtype = "f32")
  grad <- f(x)[[1L]]

  expect_equal(as_array(grad), exp(2))
})

test_that("p_reduce_max backward", {
  f <- jit(gradient(function(x) {
    rows_max <- nvl_reduce_max(x, dims = 2L, drop = TRUE)
    nv_reduce_sum(rows_max, dims = 1L, drop = TRUE)
  }))

  x <- nv_tensor(
    rbind(
      c(1, 3, 2),
      c(2, 4, 5)
    )
  )

  grads <- f(x)[[1L]]

  expect_equal(
    as_array(grads),
    rbind(
      c(0, 1, 0),
      c(0, 0, 1)
    )
  )
})

test_that("p_max on ties", {
  x <- nv_tensor(c(1, 2, 2))
  grads <- jit(gradient(\(x) nv_reduce_max(x, dims = 1)))(x)
  expect_equal(as_array(grads$x), array(c(0, 0.5, 0.5), dim = 3))
})

test_that("p_max", {
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(3, 2, 1))

  grads <- jit(gradient(\(x, y) nv_reduce_sum(nv_max(x, y), dims = 1)))(x, y)

  expect_equal(as_array(grads$x), array(c(0, 0.5, 1), dim = 3))
  expect_equal(as_array(grads$y), array(c(1, 0.5, 0), dim = 3))
})

test_that("p_min", {
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(3, 2, 1))

  grads <- jit(gradient(\(x, y) nv_reduce_sum(nv_min(x, y), dims = 1)))(x, y)

  expect_equal(as_array(grads$x), array(c(1, 0.5, 0), dim = 3))
  expect_equal(as_array(grads$y), array(c(0, 0.5, 1), dim = 3))
})

test_that("p_convert backward converts gradients to the input dtype", {
  x_arr <- array(1:6, c(2, 3))
  x <- nv_tensor(x_arr, dtype = "f32")
  f <- jit(gradient(function(x) {
    y <- nvl_convert(x, dtype = "f64")
    nv_reduce_sum(y, dims = 1:2, drop = TRUE)
  }))

  grads <- f(x)
  expect_equal(as_array(grads[[1L]]), array(1, dim = dim(x_arr)))
  expect_equal(dtype(grads[[1L]]), as_dtype("f32"))
})

test_that("p_eq, p_ne, p_gt, p_ge, p_lt, p_le", {
  a <- 1
  b <- 2

  a_nv <- nv_scalar(a)
  b_nv <- nv_scalar(b)

  comparators <- list(
    list(fun = nv_eq, expected = a == b),
    list(fun = nv_ne, expected = a != b),
    list(fun = nv_gt, expected = a > b),
    list(fun = nv_ge, expected = a >= b),
    list(fun = nv_lt, expected = a < b),
    list(fun = nv_le, expected = a <= b)
  )

  for (cmp in comparators) {
    out <- as_array(jit(cmp$fun)(a_nv, b_nv))
    expect_identical(out, cmp$expected)

    g <- jit(gradient(function(a, b) {
      cmp$fun(a, b)
    }))

    # can't compute gradients if function doesn't return
    # float
    expect_snapshot_error({
      grads <- g(a_nv, b_nv)
    })

    g <- jit(gradient(function(a, b) {
      nv_convert(cmp$fun(a, b), "f32")
    }))

    grads <- g(a_nv, b_nv)
    expect_equal(as_array(grads[[1L]]), 0)
    expect_equal(as_array(grads[[2L]]), 0)
  }
})

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "test-primitives-backward-torch.R", package = "anvil"))
}
