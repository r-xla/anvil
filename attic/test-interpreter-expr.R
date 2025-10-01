test_that("addition works", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  r2ir(
    nvl_add,
    ShapedTensor(FloatType("f32"), Shape(c(1, 1))),
    ShapedTensor(FloatType("f32"), Shape(c(1, 1)))
  )
})
