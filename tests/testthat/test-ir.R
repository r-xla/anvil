test_that("IRVariable", {
  var <- IRVariable(
    aval = ShapedTensor(FloatType("f32"), Shape(1L))
  )
  expect_class(lit_arr, "nvl_tensor")
})

test_that("IRLiteral", {

})

test_that("IREquation", {

})

test_that("IR", {

})
