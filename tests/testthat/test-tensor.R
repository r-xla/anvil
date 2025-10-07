test_that("tensor", {
  x <- nv_tensor(1:4, dtype = "i32", shape = c(4, 1))
  x
  expect_snapshot(x)
  expect_class(x, "AnvilTensor")
})

test_that("nv_scalar", {
  x <- nv_scalar(1, dtype = "f32")
  x
  expect_snapshot(x)
  expect_class(x, "AnvilTensor")
})

test_that("ShapedTensor", {
  x <- ShapedTensor(
    FloatType(32),
    Shape(c(2, 3))
  )
  expect_snapshot(x)
  expect_true(inherits(x, ShapedTensor))
  expect_true(x == x)

  expect_false(
    x ==
      ShapedTensor(
        FloatType(32),
        Shape(c(2, 1))
      )
  )

  expect_false(
    x ==
      ShapedTensor(
        FloatType(64),
        Shape(c(2, 3))
      )
  )
})

test_that("ConcreteTensor", {
  x <- ConcreteTensor(
    nv_tensor(1:6, dtype = "f32", shape = c(2, 3))
  )
  expect_true(inherits(x, ConcreteTensor))
  expect_snapshot(x)
})

test_that("from TensorDataType", {
  expect_class(nv_tensor(1L, dt_i32), "AnvilTensor")
  expect_class(nv_scalar(1L, dt_i32), "AnvilTensor")
  expect_class(nv_empty(dt_i32, c(4, 1)), "AnvilTensor")
})
