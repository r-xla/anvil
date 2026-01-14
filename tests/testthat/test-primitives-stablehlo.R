test_that("p_sine", {
  x <- nv_tensor(c(0, pi / 2, pi, 3 / 2 * pi), dtype = "f64")
  out <- as_array(jit(nvl_sine)(x))
  expect_equal(c(out), c(0, 1, 0, -1), tolerance = 1e-15)
})

test_that("p_cosine", {
  x <- nv_tensor(c(0, pi / 2, pi, 3 / 2 * pi), dtype = "f64")
  out <- as_array(jit(nvl_cosine)(x))
  expect_equal(c(out), c(1, 0, -1, 0), tolerance = 1e-15)
})

test_that("p_rng_bit_generator", {
  f <- function() {
    nvl_rng_bit_generator(nv_tensor(c(1, 2), dtype = "ui64"), "THREE_FRY", "i64", c(2, 2))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 6L))
  expect_equal(as_array(out[[2]]), array(c(43444564L, 1672743891L, -315321645L, 2109414752L), c(2, 2)))
})

test_that("p_bitcast_convert", {
  f <- function() {
    nv_bitcast_convert(
      nv_tensor(seq(-1, 1, length.out = 6), dtype = "f64", shape = c(2, 3)),
      dtype = "i32"
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(dim(as_array(out)), c(2, 3, 2))
  expect_true(is.integer(as_array(out)))
})

test_that("p_slice", {
  f <- function() {
    nv_slice(
      nv_tensor(1:6, dtype = "ui64", shape = c(2, 3)),
      start_indices = c(1, 1),
      limit_indices = c(2, 2),
      strides = c(1, 1)
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(as_array(out), matrix(c(1:4), nrow = 2))
})

test_that("p_concatenate", {
  f <- function() {
    nv_concatenate(
      nv_tensor(c(1:6), dtype = "ui64", shape = c(2, 3)),
      nv_tensor(c(7:10), dtype = "ui64", shape = c(2, 2)),
      dimension = 2L
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(dim(as_array(out)), c(2, 5))
})
test_that("p_fill", {
  f <- jit(function(x) nv_fill(x, shape = c(2, 3), dtype = "f32"), static = "x")
  expect_equal(f(1), nv_tensor(1, shape = c(2, 3), dtype = "f32"))
  expect_equal(f(2), nv_tensor(2, shape = c(2, 3), dtype = "f32"))

  # scalars
  expect_equal(
    jit(\() nv_fill(1L, shape = c(), dtype = "f32"))(),
    nv_scalar(1, dtype = "f32")
  )
  expect_equal(
    jit(\() nv_fill(1L, shape = integer(), dtype = "f32"))(),
    nv_scalar(1, dtype = "f32")
  )
  expect_equal(
    jit(\() nv_fill(1L, shape = 1L, dtype = "f32"))(),
    nv_tensor(1, shape = 1L, dtype = "f32")
  )
})

test_that("p_shift_left", {
  x <- nv_tensor(as.integer(c(1L, 2L, 3L, 8L)), dtype = "i32")
  y <- nv_tensor(as.integer(c(0L, 1L, 2L, 3L)), dtype = "i32")
  out <- as.integer(as_array(jit(nvl_shift_left)(x, y)))
  expect_equal(out, as.integer(c(1L, 4L, 12L, 64L)))
})

test_that("p_shift_right_logical", {
  x <- nv_tensor(as.integer(c(16L, 8L, 7L, 1L)), dtype = "i32")
  y <- nv_tensor(as.integer(c(0L, 1L, 2L, 0L)), dtype = "i32")
  out <- as.integer(as_array(jit(nvl_shift_right_logical)(x, y)))
  expect_equal(out, as.integer(c(16L, 4L, 1L, 1L)))
})

test_that("p_shift_right_arithmetic", {
  x <- nv_tensor(as.integer(c(-8L, -1L, 8L, -17L)), dtype = "i32")
  y <- nv_tensor(as.integer(c(1L, 3L, 2L, 4L)), dtype = "i32")
  out <- as.integer(as_array(jit(nvl_shift_right_arithmetic)(x, y)))
  expect_equal(out, as.integer(c(-4L, -1L, 2L, -2L)))
})

test_that("p_rng_bit_generator", {
  f <- function() {
    nvl_rng_bit_generator(nv_tensor(c(1, 2), dtype = "ui64"), "THREE_FRY", "i64", c(2, 2))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 6L))
  expect_equal(as_array(out[[2]]), array(c(43444564L, 1672743891L, -315321645L, 2109414752L), c(2, 2)))
})

# Reduction ops (simplified hardcoded examples, no torch comparisons)

test_that("p_reduce_sum", {
  x <- array(1:6, c(2, 3))
  f <- jit(function(a) nvl_reduce_sum(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(9, 12)))
})

test_that("p_reduce_prod", {
  x <- array(1:6, c(2, 3))
  f <- jit(function(a) nvl_reduce_prod(a, dims = 1L, drop = FALSE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(2, 12, 30), c(1, 3)))
})

test_that("p_reduce_max", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  f <- jit(function(a) nvl_reduce_max(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(0, 4)))
})

test_that("p_reduce_max drop = FALSE", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  f <- jit(function(a) nvl_reduce_max(a, dims = 2L, drop = FALSE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(0, 4), c(2, 1)))
})

test_that("p_reduce_min", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  f <- jit(function(a) nvl_reduce_min(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(-1, 2)))
})

test_that("p_reduce_min drop = FALSE", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  f <- jit(function(a) nvl_reduce_min(a, dims = 2L, drop = FALSE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(-1, 2), c(2, 1)))
})

test_that("p_reduce_any", {
  x <- array(c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE), c(2, 3))
  f <- jit(function(a) nvl_reduce_any(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "pred")))
  expect_equal(out, array(c(TRUE, FALSE)))
})

test_that("p_reduce_all", {
  x <- array(c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE), c(2, 3))
  f <- jit(function(a) nvl_reduce_all(a, dims = 1L, drop = FALSE))
  out <- as_array(f(nv_tensor(x, dtype = "pred")))
  expect_equal(out, array(rep(FALSE, 3), c(1, 3)))
})

test_that("p_broadcast_in_dim", {
  x <- 1L
  f <- jit(nvl_broadcast_in_dim, static = c("shape", "broadcast_dimensions"))
  expect_equal(
    f(nv_scalar(1L), c(1, 2), integer()),
    nv_tensor(1L, shape = c(1, 2)),
    tolerance = 1e-5
  )
})

test_that("p_reshape", {
  f <- jit(nvl_reshape, static = "shape")
  x <- array(1:6, c(3, 2))
  expect_equal(
    f(nv_tensor(x), shape = 6),
    nv_tensor(as.integer(c(1, 4, 2, 5, 3, 6)), "i32")
  )
})

test_that("p_transpose", {
  x <- array(1:4, c(2, 2))
  f <- jit(\(x) nvl_transpose(x, c(2, 1)))
  expect_equal(
    t(x),
    as_array(f(nv_tensor(x)))
  )
})

test_that("p_if: capture non-argument", {
  f <- jit(function(pred, x) {
    x1 <- nv_mul(x, x)
    x2 <- nv_add(x, x)
    nv_if(pred, x1, x2)
  })
  f(nv_scalar(TRUE), nv_scalar(2))
})

test_that("p_if", {
  # simple
  f <- function(pred, x) nvl_if(pred, x, x * x)
  fj <- jit(f)
  expect_equal(fj(nv_scalar(TRUE), nv_scalar(2)), nv_scalar(2))
  expect_equal(fj(nv_scalar(FALSE), nv_scalar(2)), nv_scalar(4))
  graph <- trace_fn(f, list(pred = nv_scalar(TRUE), x = nv_scalar(2)))

  graph <- trace_fn(f, list(pred = nv_scalar(TRUE), x = nv_scalar(2)))

  f <- jit(function(pred, x) {
    nvl_if(pred, list(list(x)), list(list(x * x)))
  })
  expect_equal(
    f(nv_scalar(TRUE), nv_scalar(2)),
    list(list(nv_scalar(2)))
  )
  expect_equal(
    f(nv_scalar(FALSE), nv_scalar(2)),
    list(list(nv_scalar(4)))
  )

  g <- jit(function(pred, x) {
    nvl_if(pred, list(x[[1]]), list(x[[1]] * x[[1]]))
  })
  expect_equal(
    g(nv_scalar(FALSE), list(nv_scalar(2))),
    list(nv_scalar(4))
  )
})

test_that("p_if: identically constants in both branches receive the same GraphValue", {
  x <- nv_scalar(1)
  f <- function(y) nvl_if(y, x, x)
  graph <- trace_fn(f, list(y = nv_scalar(TRUE)))
  fj <- jit(f)
  expect_equal(fj(nv_scalar(TRUE)), nv_scalar(1))
  expect_equal(fj(nv_scalar(FALSE)), nv_scalar(1))

  g <- jit(function(pred) {
    y <- nv_scalar(2)
    nvl_if(pred, y, y * nv_scalar(3))
  })
  expect_equal(g(nv_scalar(TRUE)), nv_scalar(2))
  expect_equal(g(nv_scalar(FALSE)), nv_scalar(6))
})

# TODO: Continue here
test_that("p_while: simple case", {
  f <- jit(function(n) {
    nv_while(list(i = nv_scalar(1L)), \(i) i <= n, \(i) {
      i <- i + nv_scalar(1L)
      list(i = i)
    })
  })

  expect_equal(
    f(nv_scalar(10L)),
    list(i = nv_scalar(11L))
  )
})

test_that("p_while: use literals in the loop", {
  f <- jit(function(n) {
    nv_while(list(i = nv_scalar(1L)), \(i) i <= n, \(i) {
      i <- i + 1L
      list(i = i)
    })
  })
  expect_equal(f(nv_scalar(10L)), list(i = nv_scalar(11L)))
})

test_that("p_while: two state variables", {
  f <- jit(function(n) {
    nv_while(
      list(i = nv_scalar(1L), s = nv_scalar(0L)),
      \(i, s) i <= n,
      \(i, s) {
        i <- i + nv_scalar(1L)
        s <- s + i
        list(i = i, s = s)
      }
    )
  })

  res <- f(nv_scalar(10L))
  expect_equal(
    res$i,
    nv_scalar(11L)
  )
  # s counts sum of i at each increment; i advances from 2 to 11
  # s = sum(2:11) = 2+3+...+11 = 65
  expect_equal(
    res$s,
    nv_scalar(sum(2:11))
  )
})

test_that("p_while: two states", {
  f <- jit(function(n) {
    nv_while(
      list(i = nv_scalar(1L), j = nv_scalar(2L)),
      \(i, j) {
        # nolint
        i <= n
      },
      \(i, j) {
        i <- i + nv_scalar(1L)
        list(i = i, j = j)
      }
    ) # nolint
  })

  expect_equal(
    f(nv_scalar(10L)),
    list(i = nv_scalar(11L), j = nv_scalar(2L))
  )
})

test_that("p_while: nested state", {
  f <- jit(function(n) {
    nv_while(
      list(i = list(nv_scalar(1L))),
      \(i) {
        i[[1]] <= n
      },
      \(i) {
        i <- i[[1L]]
        i <- i + nv_scalar(1L)
        list(i = list(i))
      }
    )
  })
  expect_equal(
    f(nv_scalar(10L)),
    list(i = list(nv_scalar(11L)))
  )
})

test_that("p_while: errors", {
  # TODO:
})


test_that("error when multiplying lists in if-statement", {
  f <- jit(function(pred, x) {
    nvl_if(pred, x + x, x * x)
  })
  expect_error(
    f(nv_scalar(FALSE), list(nv_scalar(2))),
    "non-numeric argument to binary operator"
  )
})

test_that("p_is_finite", {
  f <- jit(function(x) nvl_is_finite(x))
  x <- nv_tensor(c(1.0, Inf, -Inf, NaN), dtype = "f32")
  expect_equal(f(x), nv_tensor(c(TRUE, FALSE, FALSE, FALSE), dtype = "pred"))
})

test_that("p_clamp", {
  f <- jit(function(x) {
    min_val <- nv_broadcast_to(nv_scalar(-1.0, "f32"), shape(x))
    max_val <- nv_broadcast_to(nv_scalar(1.0, "f32"), shape(x))
    nvl_clamp(min_val, x, max_val)
  })
  x <- nv_tensor(c(-2.0, -0.5, 0.5, 2.0), dtype = "f32")
  expect_equal(f(x), nv_tensor(c(-1.0, -0.5, 0.5, 1.0), dtype = "f32"))
})

test_that("p_reverse", {
  f <- jit(function(x) nvl_reverse(x, 1L))
  x <- nv_tensor(1:5, dtype = "i32")
  expect_equal(f(x), nv_tensor(5:1, dtype = "i32"))

  # 2D reverse
  f2 <- jit(function(x) nvl_reverse(x, 2L))
  x2 <- nv_tensor(matrix(1:6, 2, 3), dtype = "i32")
  expect_equal(f2(x2), nv_tensor(matrix(c(5L, 6L, 3L, 4L, 1L, 2L), 2, 3), dtype = "i32"))
})

test_that("p_iota", {
  f <- jit(function() nvl_iota(1L, "i32", 5L))
  expect_equal(f(), nv_tensor(0:4, dtype = "i32"))

  # 2D along first dimension
  f2 <- jit(function() nvl_iota(1L, "i32", c(3L, 2L)))
  expected <- matrix(c(0L, 1L, 2L, 0L, 1L, 2L), 3, 2)
  expect_equal(f2(), nv_tensor(expected, dtype = "i32"))
})

test_that("p_popcnt", {
  f <- jit(function(x) nvl_popcnt(x))
  x <- nv_tensor(c(0L, 1L, 2L, 3L, 7L, 255L), dtype = "i32")
  expect_equal(f(x), nv_tensor(c(0L, 1L, 1L, 2L, 3L, 8L), dtype = "i32"))
})

test_that("p_print", {
  skip_if(!is_cpu(), "print_tensor only works on CPU")

  f <- jit(function(x) nvl_print(x))
  x <- nv_tensor(c(1.0, 2.0, 3.0), dtype = "f32")
  out <- f(x)
  expect_equal(as_array(out), as_array(x))
  expect_snapshot(f(x))
})

# we don't want to include torch in Suggests just for the tests, as it's a relatively
# heavy dependency
# We have a CI job that installs torch, so it's at least tested once

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "test-primitives-stablehlo-torch.R", package = "anvil"))
}
