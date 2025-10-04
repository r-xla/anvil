test_that("simple function works (scalar)", {
  f_grad <- jit(gradient(nvl_mul))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(2.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("chain rule works (scalar)", {
  f <- function(x, y) {
    nvl_add(nvl_mul(x, y), x)
  }

  f_grad <- jit(gradient(f))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(3.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("gradient does not have to depend on input", {
  # This is special, because the input has no influence
  # on the gradient, because the gradient is constant
  f <- function(x, y) {
    nvl_add(x, y)
  }

  f_grad <- jit(gradient(f))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("nested inputs", {
  f <- jit(gradient(function(x) {
    nvl_mul(x[[1]][[1]], x[[1]][[1]])
  }))
  expect_equal(f(list(list(nv_scalar(1))))[[1L]], list(list(nv_scalar(2))))
})

test_that("no nested outpus", {
  # we expect a scalar output
  # -> check for good error message
})

test_that("constants work (scalar)", {
  f <- jit(gradient(function(x) {
    nvl_mul(x, nv_scalar(2))
  }))
  expect_equal(f(nv_scalar(1))[[1L]], nv_scalar(2))
})

test_that("broadcasting works", {
  # TODO
})

test_that("second order gradient (scalar)", {
  # this works only for scalar functions, so this is primarily a stress
  # test for the interpreter as it is not that useful
  f <- function(x) {
    nvl_mul(x, x)
  }
  fg2 <- jit(gradient(\(x) gradient(f)(x)[[1L]]))
  fg2(nv_scalar(1))

  expect_equal(fg2(nv_scalar(1)), list(x = nv_scalar(2)))
})

test_that("neg works", {
  g <- jit(gradient(nvl_neg))
  expect_equal(g(nv_scalar(1))[[1L]], nv_scalar(-1))
})

test_that("names for grad: primitive", {
  g <- jit(gradient(`*`))
  expect_equal(formalArgs2(g), c("e1", "e2"))
  expect_equal(
    list(
      e1 = nv_scalar(1),
      e2 = nv_scalar(2)
    ),
    g(nv_scalar(2), nv_scalar(1))
  )
})

test_that("names for grad: function", {
  f <- function(e1, e2) {
    e1 * e2
  }
  g <- jit(gradient(f))
  expect_equal(formals(g), formals(f))
  expect_equal(
    list(
      e1 = nv_scalar(1),
      e2 = nv_scalar(2)
    ),
    g(nv_scalar(2), nv_scalar(1))
  )
})

# New tests for selective gradients (wrt)

test_that("partial gradient simple", {
  f <- function(lhs, rhs) {
    nvl_add(lhs, rhs)
  }
  g <- jit(gradient(f, wrt = "lhs"))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))[[1L]]
  expect_equal(out, nv_scalar(1.0))
})

#test_that("partial gradient: y = a * (x * b) wrt x", {
#  f <- function(a, x, b) {
#    a * (x * b)
#  }
#  g <- gradient(f, wrt = "x")
#  a <- nv_scalar(2.0)
#  x <- nv_scalar(3.0)
#  b <- nv_scalar(5.0)
#  out <- g(a, x, b)
#  expect_null(out[[1L]])
#  expect_equal(out[[2L]], nv_scalar(10.0))
#  expect_null(out[[3L]])
#})
#
#test_that("wrt nested inputs", {
#  # TODO:
#})
#
