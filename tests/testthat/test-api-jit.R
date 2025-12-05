test_that("generate state", {
  f <- function() {
    nv_runif(
      initial_state = nv_generate_state(1),
      dtype = "f64",
      shape_out = c(10, 5)
    )
  }
  g <- jit(f)
  out1 <- g()

  f <- function() {
    nv_runif(
      initial_state = nv_generate_state(1),
      dtype = "f64",
      shape_out = c(10, 5)
    )
  }
  g <- jit(f)
  out2 <- g()

  # check resulting states
  expect_equal(as_array(out1[[1]]), as_array(out2[[1]]))
  # check random variables
  expect_equal(as_array(out1[[2]]), as_array(out2[[2]]))
})

test_that("seed2state", {
  # auto-detect state
  set.seed(42)
  f <- function() {
    nv_seed2state(shape_out = c(3, 2))
  }
  g <- jit(f)
  out1 <- g()

  # explicitly provide state
  set.seed(42)
  f <- function() {
    nv_seed2state(shape_out = c(3, 2), random_seed = .Random.seed)
  }
  g <- jit(f)
  out2 <- g()

  expect_true(identical(as_array(out1), as_array(out2)))
  expect_equal(shape(out2), c(3, 2))
  expect_true(inherits(dtype.AnvilTensor(out1), UnsignedType))
  expect_equal(dtype.AnvilTensor(out1)@value, 64L)

  # test ui32
  set.seed(1)
  f <- function() {
    nv_seed2state(shape_out = 2, dtype = "ui32")
  }
  g <- jit(f)
  out3 <- g()
  expect_equal(shape(out3), 2)
  expect_true(inherits(dtype.AnvilTensor(out3), UnsignedType))
  expect_equal(dtype.AnvilTensor(out3)@value, 32L)
})


test_that("p_rnorm", {
  # basic test
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape_out = c(2, 3))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 6L))
  expect_true(inherits(dtype.AnvilTensor(out[[2]]), FloatType))
  expect_equal(dtype.AnvilTensor(out[[2]])@value, 32L)

  # test with uneven total number of RVs
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape_out = c(3, 3))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 8L))
  expect_equal(shape(out[[2]]), c(3L, 3L))

  # check normality
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f64", shape_out = c(200L, 300L, 400L))
  }
  g <- jit(f)
  out <- g()
  Z <- as_array(out[[2]])
  expect_false(any(!is.finite(Z)))

  tst <- apply(Z, c(2, 3), \(z) shapiro.test(z)$p.value)
  expect_equal(round(mean(c(tst) < .05), 2), .05)

  f <- function() {
    nv_rnorm(
      nv_tensor(c(3, 83), dtype = "ui64"),
      dtype = "f64",
      shape_out = c(10L, 10L, 10L, 10L, 10L),
      mu = 10,
      sigma = 9
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(round(mean(as_array(out[[2]])), 1), 10)
  expect_equal(round(sd(as_array(out[[2]])), 1), 9)
})

test_that("p_runif", {
  f <- function() {
    nv_runif(
      nv_tensor(c(1, 2), dtype = "ui64"),
      dtype = "f32",
      shape_out = c(10, 20, 30, 40, 50),
      lower = -1,
      upper = 1
    )
  }
  g <- jit(f)
  out <- g()

  expect_false(any(as_array(out[[2]]) == -1))
  expect_false(any(as_array(out[[2]]) == 1))
  expect_equal(mean(as_array(out[[2]])), 0, tolerance = 1e-3)
  expect_equal(var(as_array(out[[2]])), 1 / 3, tolerance = 1e-3)

  expect_equal(c(as_array(out[[1]])), c(1L, 6000002L))
  expect_equal(
    shape(out[[2]]),
    c(10, 20, 30, 40, 50)
  )
})


test_that("auto-broadcasting higher-dimensional tensors is not supported (it's bug prone)", {
  x <- nv_tensor(1:2, shape = c(2, 1))
  y <- nv_tensor(1:2, shape = c(1, 2))
  expect_error(
    nv_add(x, y),
    "By default, only scalar broadcasting is supported"
  )
})

test_that("broadcasting scalars", {
  fjit <- jit(nv_add)
  expect_equal(
    fjit(
      nv_scalar(1),
      nv_tensor(0, shape = c(2, 2))
    ),
    nv_tensor(1, shape = c(2, 2))
  )
})

test_that("infix add", {
  f <- jit(function(x, y) {
    x + y
  })
  expect_equal(
    f(
      nv_tensor(1, shape = c(2, 2)),
      nv_tensor(0, shape = c(2, 2))
    ),
    nv_tensor(1, shape = c(2, 2))
  )
})

test_that("jit constant single return is bare tensor", {
  f <- jit(function() nv_scalar(0.5))
  out <- f()
  expect_equal(as_array(out), 0.5, tolerance = 1e-6)
})

test_that("Summary group generics", {
  fsum <- jit(function(x) sum(x))
  expect_equal(as_array(fsum(nv_tensor(1:10))), 55)
})

test_that("mean", {
  fmean <- jit(function(x) mean(x))
  expect_equal(as_array(fmean(nv_tensor(1:10, "f32"))), 5.5)
})

test_that("constants can be lifted to the appropriate level", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  jit(gradient(f, wrt = "x"))(nv_scalar(2))
})

test_that("wrt non-existent argument", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_error(
    jit(gradient(f, wrt = "y"))(nv_tensor(2)),
    "must be a subset"
  )
})
