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

test_that("p_reduce_min backward", {
  f <- jit(gradient(function(x) {
    rows_min <- nvl_reduce_min(x, dims = 2L, drop = TRUE)
    nv_reduce_sum(rows_min, dims = 1L, drop = TRUE)
  }))

  x <- nv_tensor(
    rbind(
      c(2, 4, 5),
      c(1, 1, 9)
    ),
    dtype = "f32",
    shape = c(2, 3)
  )

  grads <- f(x)[[1L]]

  expect_equal(
    as_array(grads),
    rbind(
      c(1, 0, 0),
      c(0.5, 0.5, 0)
    ),
    tolerance = 1e-6
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
    y <- nvl_convert(x, dtype = "f64", ambiguous = FALSE)
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

test_that("p_pad backward with interior padding", {
  # Interior padding adds padding between elements
  # For input [a, b, c] with interior_padding=1, output is [a, 0, b, 0, c]
  # Gradient flows only to original positions [a, b, c]
  f <- jit(gradient(function(x) {
    x <- x * nv_tensor(c(1, 2, 3), dtype = "f64")
    y <- nvl_pad(x, nv_scalar(0, "f64"), 0L, 0L, 1L)
    nv_reduce_sum(y, dims = 1L, drop = TRUE)
  }))
  x <- nv_tensor(c(1, 2, 3), dtype = "f64")
  g <- f(x)
  expect_equal(g[[1L]], nv_tensor(1:3, dtype = "f64"))

  # Test with both edge and interior padding
  f2 <- jit(gradient(function(x) {
    x <- x * nv_tensor(c(1, 2), dtype = "f64")
    # edge_padding_low=1, edge_padding_high=1, interior_padding=1
    # For input [a, b], output is [0, a, 0, b, 0]
    y <- nvl_pad(x, nv_scalar(0, "f64"), 1L, 1L, 1L)
    nv_reduce_sum(y, dims = 1L, drop = TRUE)
  }))
  x2 <- nv_tensor(c(5, 10), dtype = "f64")
  g2 <- f2(x2)
  expect_equal(g2[[1L]], nv_tensor(c(1, 2), dtype = "f64"))

  # Test 2D with interior padding
  f3 <- jit(gradient(function(x) {
    x <- x * nv_tensor(c(1, 2, 3, 4), shape = c(2, 2), dtype = "f64")
    y <- nvl_pad(x, nv_scalar(0, "f64"), c(0L, 0L), c(0L, 0L), c(1L, 1L))
    nv_reduce_sum(y, dims = c(1L, 2L), drop = TRUE)
  }))
  x3 <- nv_tensor(matrix(1:4, 2, 2), dtype = "f64")
  g3 <- f3(x3)
  expect_equal(g3[[1L]], nv_tensor(matrix(1:4, 2, 2), dtype = "f64"))

  # Test 2D with different edge padding on each dimension
  f4 <- jit(gradient(function(x) {
    y <- nvl_pad(x, nv_scalar(0, "f64"), c(1L, 2L), c(2L, 1L), c(0L, 0L))
    nv_reduce_sum(y, dims = c(1L, 2L), drop = TRUE)
  }))
  x4 <- nv_tensor(matrix(1:6, 2, 3), dtype = "f64")
  g4 <- f4(x4)
  expect_equal(g4[[1L]], nv_tensor(matrix(rep(1, 6), 2, 3), dtype = "f64"))
})

test_that("p_dynamic_slice backward", {
  # Gradient should scatter the incoming gradient back to the operand position
  f <- jit(gradient(
    function(x, start_i) {
      sliced <- nvl_dynamic_slice(x, start_i, slice_sizes = 3L)
      nv_reduce_sum(sliced, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_tensor(1:10, dtype = "f64", shape = 10L)
  start_i <- nv_scalar(3L, dtype = "i32")
  grad <- f(x, start_i)[[1L]]

  # Gradient is 1 at positions 3, 4, 5 (the sliced region) and 0 elsewhere
  expect_equal(grad, nv_tensor(c(0, 0, 1, 1, 1, 0, 0, 0, 0, 0), dtype = "f64", shape = 10L))

  # Test 2D case
  f2d <- jit(gradient(
    function(x, start_i, start_j) {
      sliced <- nvl_dynamic_slice(x, start_i, start_j, slice_sizes = c(2L, 2L))
      nv_reduce_sum(sliced, dims = c(1L, 2L), drop = TRUE)
    },
    wrt = "x"
  ))

  x2d <- nv_tensor(1:12, dtype = "f64", shape = c(3L, 4L))
  start_i <- nv_scalar(2L, dtype = "i32")
  start_j <- nv_scalar(2L, dtype = "i32")
  grad2d <- f2d(x2d, start_i, start_j)[[1L]]

  # Gradient is 1 at positions (2,2), (2,3), (3,2), (3,3) and 0 elsewhere
  expect_equal(grad2d, nv_tensor(c(0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0), dtype = "f64", shape = c(3, 4)))
})

test_that("p_dynamic_slice backward with out-of-bounds", {
  # Test that gradient works correctly when indices are clamped
  f <- jit(gradient(
    function(x, start_i) {
      sliced <- nvl_dynamic_slice(x, start_i, slice_sizes = c(5L))
      nv_reduce_sum(sliced, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_tensor(1:10, dtype = "f64", shape = 10L)
  # Request slice at position 8 with size 5, will be clamped to position 6
  start_i <- nv_scalar(8L, dtype = "i32")
  grad <- f(x, start_i)[[1L]]

  # Gradient should be 1 at positions 6-10 (the actual sliced region after clamping)
  expect_equal(grad, nv_tensor(c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1), dtype = "f64", shape = 10L))
})

test_that("p_dynamic_update_slice backward", {
  # Test gradient for operand (zero out the updated region)
  f_operand <- jit(gradient(
    function(x, update, start_i) {
      updated <- nvl_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_tensor(1:10, dtype = "f64", shape = 10L)
  update <- nv_tensor(c(100, 200, 300), dtype = "f64", shape = 3L)
  start_i <- nv_scalar(4L, dtype = "i32")
  grad_x <- f_operand(x, update, start_i)[[1L]]

  # Gradient is 0 at positions 4, 5, 6 (the updated region) and 1 elsewhere
  expect_equal(grad_x, nv_tensor(c(1, 1, 1, 0, 0, 0, 1, 1, 1, 1), dtype = "f64", shape = 10L))

  # Test gradient for update (slice out the corresponding region from grad)
  f_update <- jit(gradient(
    function(x, update, start_i) {
      updated <- nvl_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "update"
  ))

  grad_update <- f_update(x, update, start_i)[[1L]]

  # Gradient for update is all 1s (since we sum everything)
  expect_equal(grad_update, nv_tensor(c(1, 1, 1), dtype = "f64", shape = 3L))
})

test_that("p_dynamic_update_slice backward with out-of-bounds", {
  # Test that gradient works correctly when indices are clamped
  f_operand <- jit(gradient(
    function(x, update, start_i) {
      updated <- nvl_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_tensor(1:10, dtype = "f64", shape = 10L)
  update <- nv_tensor(c(100, 200, 300, 400, 500), dtype = "f64", shape = 5L)
  # Request update at position 8 with size 5, will be clamped to position 6
  start_i <- nv_scalar(8L, dtype = "i32")
  grad_x <- f_operand(x, update, start_i)[[1L]]

  # Gradient is 0 at positions 6-10 (the actual updated region after clamping) and 1 elsewhere
  expect_equal(grad_x, nv_tensor(c(1, 1, 1, 1, 1, 0, 0, 0, 0, 0), dtype = "f64", shape = 10L))

  # Test gradient for update with out-of-bounds
  f_update <- jit(gradient(
    function(x, update, start_i) {
      updated <- nvl_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "update"
  ))

  grad_update <- f_update(x, update, start_i)[[1L]]

  # Gradient for update is all 1s (since we sum everything from the clamped position)
  expect_equal(grad_update, nv_tensor(c(1, 1, 1, 1, 1), dtype = "f64", shape = 5L))
})

test_that("p_is_finite", {
  x <- nv_tensor(c(1.0, Inf, -Inf, NaN, 0.5, -0.5), dtype = "f32")
  verify_zero_grad_unary(nvl_is_finite, x)
})

test_that("p_popcnt", {
  x <- nv_tensor(c(0L, 1L, 3L, 7L, 15L), dtype = "i32")
  verify_zero_grad_unary(nvl_popcnt, x)
})

describe("shift ops", {
  it("p_shift_left returns zero gradients", {
    x <- nv_tensor(c(1L, 2L, 4L, 8L), dtype = "i32")
    y <- nv_tensor(c(1L, 1L, 1L, 1L), dtype = "i32")
    verify_zero_grad_binary(nvl_shift_left, x, y)
  })

  it("p_shift_right_arithmetic returns zero gradients", {
    x <- nv_tensor(c(8L, 16L, 32L, -8L), dtype = "i32")
    y <- nv_tensor(c(1L, 2L, 1L, 1L), dtype = "i32")
    verify_zero_grad_binary(nvl_shift_right_arithmetic, x, y)
  })

  it("p_shift_right_logical returns zero gradients", {
    x <- nv_tensor(c(8L, 16L, 32L, 64L), dtype = "i32")
    y <- nv_tensor(c(1L, 2L, 1L, 2L), dtype = "i32")
    verify_zero_grad_binary(nvl_shift_right_logical, x, y)
  })
})

test_that("p_bitcast_convert", {
  x <- nv_tensor(c(1.0, 2.0, 3.0, 4.0), dtype = "f32")
  verify_zero_grad_unary(nvl_bitcast_convert, x, f_wrapper = function(x) {
    out <- nvl_bitcast_convert(x, dtype = "i32")
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  })
})

describe("boolean ops", {
  expected_zeros <- nv_tensor(rep(0, 4), dtype = "f32")

  verify_bool_binary <- function(nvl_fn) {
    x <- nv_tensor(c(1, 0, 0, 1))
    y <- nv_tensor(c(1, 1, 0, 0))
    f <- function(x, y) {
      x <- nv_convert(x, "i1")
      y <- nv_convert(y, "i1")
      out <- nvl_fn(x, y)
      nv_convert(out, "f32")
    }
    verify_zero_grad_binary(f, x, y)
  }

  verify_bool_reduce <- function(nvl_fn) {
    x <- nv_tensor(c(1.0, 1.0, 0.0, 0.0), dtype = "f32")
    f <- function(x) {
      x_pred <- nv_convert(x, "i1")
      out <- nvl_fn(x_pred, dims = 1L, drop = TRUE)
      nv_convert(out, "f32")
    }
    grads <- jit(gradient(f))(x)
    testthat::expect_equal(grads[[1L]], expected_zeros)
  }

  it("p_and returns zero gradients", {
    verify_bool_binary(nvl_and)
  })
  it("p_or returns zero gradients", {
    verify_bool_binary(nvl_or)
  })
  it("p_xor returns zero gradients", {
    verify_bool_binary(nvl_xor)
  })
  it("p_not returns zero gradients", {
    x <- nv_tensor(c(1.0, 0.0, 1.0, 0.0), dtype = "f32")
    f <- function(x) {
      x_pred <- nv_convert(x, "i1")
      out <- nvl_not(x_pred)
      out <- nv_convert(out, "f32")
      nv_reduce_sum(out, dims = 1L, drop = TRUE)
    }
    verify_zero_grad_unary(nvl_not, x, f_wrapper = f)
  })
  it("p_reduce_all returns zero gradients", {
    verify_bool_reduce(nvl_reduce_all)
  })
  it("p_reduce_any returns zero gradients", {
    verify_bool_reduce(nvl_reduce_any)
  })
})

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "test-primitives-backward-torch.R", package = "anvil"))
}
