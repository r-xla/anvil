test_that("addition works", {
  f <- function(x, y) {
    prim_add(x, y)
  }
  lower(prim_add,
    ShapedTensor(FloatType("f32"), Shape(c(1, 1))),
    ShapedTensor(FloatType("f32"), Shape(c(1, 1)))
 )
})
