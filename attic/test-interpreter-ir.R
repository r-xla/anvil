test_that("addition works", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  res <- r2ir(
    f,
    ShapedTensor(FloatType("f32"), Shape(1L)),
    ShapedTensor(FloatType("f32"), Shape(1L))
  )
  expect_snapshot(res$ir)
  expect_equal(res$consts, list())
})

test_that("constants work", {
  # IRs are pure, i.e. all constants become arguments of the IR expr
  f <- function(x) {
    nvl_add(x, nv_tensor(1.0))
  }
  res <- r2ir(f, ShapedTensor(FloatType("f32"), Shape(1L)))
  expect_snapshot(res$ir)
  expect_equal(res$consts[[1L]], nv_tensor(1.0))
})

test_that("can capture constant from outer scope", {
  x <- nv_tensor(1.0)
  f <- function(y) {
    nvl_add(y, x)
  }
  res <- r2ir(f, ShapedTensor(FloatType("f32"), Shape(1L)))
  expect_snapshot(res$ir)
  expect_equal(res$consts[[1L]], nv_tensor(1.0))
})
