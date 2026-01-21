test_that("shape2string", {
  expect_equal(shape2string(c(2, 2)), "(2,2)")
  expect_equal(shape2string(c(2, 2), parenthesize = FALSE), "2,2")
  expect_equal(shape2string(Shape(c(2, 2))), "(2,2)")
  expect_equal(shape2string(Shape(c()), parenthesize = TRUE), "()")
  expect_equal(shape2string(Shape(c()), parenthesize = FALSE), "")
})

test_that("dtype2string", {
  expect_equal(dtype2string(as_dtype("f32")), "f32")
  expect_equal(dtype2string(as_dtype("i32")), "i32")
  expect_equal(dtype2string(as_dtype("f32"), ambiguous = TRUE), "f32?")
})

test_that("dtype", {
  expect_equal(
    dtype_abstract(1L),
    as_dtype("i32")
  )
  expect_equal(
    dtype(nv_scalar(1L, dtype = "f32")),
    as_dtype("f32")
  )
})

test_that("ndims", {
  expect_equal(ndims_abstract(1L), 0L)
  expect_equal(ndims(nv_tensor(1:4, dtype = "f32", shape = c(2, 2))), 2L)
})

test_that("shape", {
  expect_equal(shape_abstract(1L), integer())
  expect_equal(shape(nv_tensor(1:4, dtype = "f32", shape = c(2, 2))), c(2, 2))
})

test_that("ambiguous", {
  # Ambiguous scalar (no explicit dtype)
  expect_true(ambiguous(nv_scalar(1.0)))
  expect_true(ambiguous(nv_scalar(1L)))

  # Non-ambiguous scalar (explicit dtype)
  expect_false(ambiguous(nv_scalar(1.0, dtype = "f32")))
  expect_false(ambiguous(nv_scalar(1L, dtype = "i32")))

  # Non-ambiguous tensor
  expect_false(ambiguous(nv_tensor(1:4, dtype = "f32", shape = c(2, 2))))

  # Primitive R values (converted via to_abstract)
  expect_true(ambiguous_abstract(1.0))
  expect_true(ambiguous_abstract(1L))
  expect_false(ambiguous_abstract(TRUE))
})
