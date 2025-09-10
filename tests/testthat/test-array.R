test_that("tensor", {
  x <- nvl_tensor(1:4, dtype = "i32", shape = c(4, 1))
  x
  expect_snapshot(x)
  expect_class(x, "nvl_tensor")
})

test_that("nvl_scalar", {
  x <- nvl_scalar(1, dtype = "f32")
  x
  expect_snapshot(x)
  expect_class(x, "nvl_tensor")
})

test_that("ShapedTensor", {
  x <- ShapedTensor(
    FloatType("f32"),
    Shape(c(2, 3))
  )
  expect_snapshot(x)
  expect_true(inherits(x, ShapedTensor))
  expect_true(x == x)

  expect_false(
    x ==
      ShapedTensor(
        FloatType("f32"),
        Shape(c(2, 1))
      )
  )

  expect_false(
    x ==
      ShapedTensor(
        FloatType("f64"),
        Shape(c(2, 3))
      )
  )
})

test_that("ConcreteTensor", {
  x <- ConcreteTensor(
    nvl_tensor(1:6, dtype = "f32", shape = c(2, 3))
  )
  expect_true(inherits(x, ConcreteTensor))
  expect_snapshot(x)
})
