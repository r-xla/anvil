test_that("array", {
  x <- nvl_array(1:4, dtype = "i32", shape = c(4, 1))
  x
  expect_snapshot(x)
  expect_class(x, "nvl_array")
})

test_that("nvl_scalar", {
  x <- nvl_scalar(1, dtype = "f32")
  x
  expect_snapshot(x)
  expect_class(x, "nvl_array")
})

test_that("ShapedArray", {
  x <- ShapedArray(
    FloatType("f32"),
    Shape(c(2, 3))
  )
  expect_snapshot(x)
  expect_true(inherits(x, ShapedArray))
  expect_true(x == x)

  expect_false(
    x ==
      ShapedArray(
        FloatType("f32"),
        Shape(c(2, 1))
      )
  )

  expect_false(
    x ==
      ShapedArray(
        FloatType("f64"),
        Shape(c(2, 3))
      )
  )
})

test_that("ConcreteArray", {
  x <- ConcreteArray(
    nvl_array(1:6, dtype = "f32", shape = c(2, 3))
  )
  expect_true(inherits(x, ConcreteArray))
  expect_snapshot(x)
})
