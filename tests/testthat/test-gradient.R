test_that("gradient: simple example", {
  f <- function(x, y) {
    nvl_mul(x, y)
  }
  g <- jit(gradient(f))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(2.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("gradient: does not depend on input", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  g <- jit(gradient(f))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})
