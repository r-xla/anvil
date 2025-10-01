test_that("builder works", {
  builder <- Builder$new()
  ip <- IRInterpreter(
    MainInterpreter(1L, IRInterpreter, builder)
  )

  expect_class(builder, "Builder")

  aval1 <- ShapedTensor(FloatType("f32"), Shape(integer()))
  aval2 <- ShapedTensor(FloatType("f32"), Shape(integer()))
  x1 <- new_arg(ip, aval1)
  expect_equal(length(builder$boxes), 1L)
  x2 <- new_arg(ip, aval2)
  expect_equal(length(builder$boxes), 2L)

  expect_equal(builder$boxes_to_variables[[ir_id(x1)]]@aval, aval1)
  expect_equal(builder$boxes_to_variables[[ir_id(x2)]]@aval, aval2)

  expect_equal(length(builder$const_boxes), 0L)
  expect_equal(length(builder$const_variables), 0L)

  # box a constant
  aval3 <- nv_tensor(1L)
  x3 <- box(ip, aval3)
  expect_equal(length(builder$boxes), 3L)
  expect_equal(length(builder$const_boxes), 1L)
  expect_equal(length(builder$const_values), 1L)

  expect_equal(
    builder$const_boxes[[ir_id(aval3)]],
    x3
  )

  expect_error(builder$build(list(x1, x2), list(x3)), NA)
})
