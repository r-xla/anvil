test_that("tensor", {
  x <- nv_tensor(1:4, dtype = "i32", shape = c(4, 1), device = "cpu")
  expect_snapshot(x)
  expect_class(x, "AnvilTensor")
  expect_equal(shape(x), c(4, 1))
  expect_equal(dtype(x), dt_i32)
  expect_equal(as_array(x), array(1:4, c(4, 1)))
})

test_that("nv_scalar", {
  x <- nv_scalar(1L, dtype = "f32", device = "cpu")
  x
  expect_class(x, "AnvilTensor")
  expect_snapshot(x)
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
    nv_tensor(1:6, dtype = "f32", shape = c(2, 3), device = "cpu")
  )
  expect_true(inherits(x, ConcreteTensor))
  expect_snapshot(x)
})

test_that("from TensorDataType", {
  expect_class(nv_tensor(1L, dt_i32), "AnvilTensor")
  expect_class(nv_scalar(1L, dt_i32), "AnvilTensor")
  expect_class(nv_empty(dt_i32, c(0, 1)), "AnvilTensor")
})

test_that("nv_tensor from nv_tensor", {
  skip_if(!is_cuda())
  x <- nv_tensor(1, device = "cuda")
  expect_equal(platform(nv_tensor(x)), "cuda")
  expect_error(nv_tensor(x, device = "cpu"))
  expect_error(nv_tensor(x, shape = c(1, 1)))
  expect_error(nv_tensor(x, dtype = "f64"))
})
