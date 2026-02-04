test_that("tensor", {
  x <- nv_tensor(1:4, dtype = "i32", shape = c(4, 1), device = "cpu")
  expect_snapshot(x)
  expect_class(x, "AnvilTensor")
  expect_equal(shape(x), c(4, 1))
  expect_equal(dtype(x), as_dtype("i32"))
  expect_equal(as_array(x), array(1:4, c(4, 1)))
})

test_that("device returns the pjrt device", {
  x <- nv_tensor(1, device = "cpu")
  expect_true(device(x) == pjrt::as_pjrt_device("cpu"))
})

test_that("nv_scalar", {
  x <- nv_scalar(1L, dtype = "f32", device = "cpu")
  x
  expect_class(x, "AnvilTensor")
  expect_snapshot(x)
})

test_that("AbstractTensor", {
  x <- AbstractTensor(
    FloatType(32),
    Shape(c(2, 3))
  )
  expect_snapshot(x)
  expect_true(inherits(x, "AbstractTensor"))
  expect_true(eq_type(x, x, ambiguity = TRUE))

  expect_false(
    eq_type(
      x,
      AbstractTensor(
        FloatType(32),
        Shape(c(2, 1))
      ),
      ambiguity = TRUE
    )
  )

  expect_false(
    eq_type(
      x,
      AbstractTensor(
        FloatType(64),
        Shape(c(2, 3))
      ),
      ambiguity = TRUE
    )
  )
})

test_that("ConcreteTensor", {
  x <- ConcreteTensor(
    nv_tensor(1:6, dtype = "f32", shape = c(2, 3), device = "cpu")
  )
  expect_true(inherits(x, "ConcreteTensor"))
  expect_snapshot(x)
})

test_that("from TensorDataType", {
  expect_class(nv_tensor(1L, "i32"), "AnvilTensor")
  expect_class(nv_scalar(1L, "i32"), "AnvilTensor")
  expect_class(nv_empty("i32", c(0, 1)), "AnvilTensor")
})

test_that("nv_tensor from nv_tensor", {
  skip_if(!is_cuda())
  x <- nv_tensor(1, device = "cuda")
  expect_equal(platform(nv_tensor(x)), "cuda")
  expect_error(nv_tensor(x, device = "cpu"))
  expect_error(nv_tensor(x, shape = c(1, 1)))
  expect_error(nv_tensor(x, dtype = "f64"))
})

test_that("format", {
  expect_equal(format(nv_tensor(1:4, shape = c(4, 1))), "AnvilTensor(dtype=i32, shape=4x1)")
})

test_that("eq_type and neq_type respect ambiguity argument", {
  # With ambiguity = TRUE, different ambiguity means not equal
  expect_true(
    neq_type(AbstractTensor("f32", 1L, TRUE), AbstractTensor("f32", 1L, FALSE), ambiguity = TRUE)
  )
  expect_true(
    neq_type(AbstractTensor("f32", 1L, FALSE), AbstractTensor("f32", 1L, TRUE), ambiguity = TRUE)
  )
  expect_true(
    eq_type(AbstractTensor("f32", 1L, TRUE), AbstractTensor("f32", 1L, TRUE), ambiguity = TRUE)
  )
  # With ambiguity = FALSE, ambiguity is ignored
  expect_true(
    eq_type(AbstractTensor("f32", 1L, TRUE), AbstractTensor("f32", 1L, FALSE), ambiguity = FALSE)
  )
  expect_true(
    eq_type(AbstractTensor("f32", 1L, FALSE), AbstractTensor("f32", 1L, TRUE), ambiguity = FALSE)
  )
})

test_that("== and != operators throw errors for AbstractTensor", {
  x <- AbstractTensor("f32", 1L, FALSE)
  y <- AbstractTensor("f32", 1L, FALSE)
  expect_error(x == y, "Use.*eq_type")
  expect_error(x != y, "Use.*neq_type")
})

test_that("to_abstract", {
  # literal
  expect_equal(to_abstract(TRUE), LiteralTensor(TRUE, c(), "pred", FALSE))
  expect_equal(to_abstract(1L), LiteralTensor(1L, c(), "i32", TRUE))
  expect_equal(to_abstract(1.0), LiteralTensor(1.0, c(), "f32", TRUE))
  # anvil tensor
  x <- nv_tensor(1:4, dtype = "f32", shape = c(2, 2))
  expect_equal(to_abstract(x), ConcreteTensor(x))
  # graph box
  aval <- GraphValue(AbstractTensor("f32", c(2, 2), FALSE))
  x <- GraphBox(aval, local_descriptor())
  expect_equal(to_abstract(x), aval$aval)

  # pure
  x <- nv_scalar(1)
  expect_equal(to_abstract(x, pure = TRUE), AbstractTensor("f32", c(), FALSE))
  expect_error(to_abstract(list(1, 2)), "is not a tensor-like object")
})


test_that("as_shape for c() (i.e., NULL)", {
  expect_equal(as_shape(c()), Shape(integer()))
})

test_that("AbstractTensor can be created with any ambiguous dtype", {
  expect_true(ambiguous(AbstractTensor("i16", integer(), TRUE)))
})

test_that("nv_aten creates AbstractTensor", {
  expect_equal(
    nv_aten("f32", c()),
    AbstractTensor("f32", Shape(integer()), FALSE)
  )
  expect_equal(
    nv_aten(as_dtype("i32"), 1:2),
    AbstractTensor("i32", Shape(1:2), FALSE)
  )
})

test_that("as_shape for c() (i.e., NULL)", {
  expect_equal(as_shape(c()), Shape(integer()))
})


test_that("to_abstract", {
  # literal
  expect_equal(to_abstract(TRUE), LiteralTensor(TRUE, c(), "pred", FALSE))
  expect_equal(to_abstract(1L), LiteralTensor(1L, c(), "i32", TRUE))
  expect_equal(to_abstract(1.0), LiteralTensor(1.0, c(), "f32", TRUE))
  # anvil tensor
  x <- nv_tensor(1:4, dtype = "f32", shape = c(2, 2))
  expect_equal(to_abstract(x), ConcreteTensor(x))
  # graph box
  aval <- GraphValue(AbstractTensor("f32", c(2, 2), FALSE))
  x <- GraphBox(aval, local_descriptor())
  expect_equal(to_abstract(x), aval$aval)
})

test_that("stablehlo dtype is printed", {
  skip_if(!is_cpu())
  expect_snapshot(nv_tensor(TRUE))
})
