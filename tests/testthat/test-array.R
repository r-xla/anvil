test_that("array", {
  x <- nv_array(1:4, dtype = "i32", shape = c(4, 1), device = "cpu")
  expect_snapshot(x)
  expect_class(x, "AnvilArray")
  expect_equal(shape(x), c(4, 1))
  expect_equal(dtype(x), as_dtype("i32"))
  expect_equal(as_array(x), array(1:4, c(4, 1)))
})

test_that("device returns the pjrt device", {
  x <- nv_array(1, device = "cpu")
  expect_true(device(x) == pjrt::as_pjrt_device("cpu"))
})

test_that("nv_scalar", {
  x <- nv_scalar(1L, dtype = "f32", device = "cpu")
  x
  expect_class(x, "AnvilArray")
  expect_snapshot(x)
})

test_that("AbstractArray", {
  x <- AbstractArray(
    FloatType(32),
    Shape(c(2, 3))
  )
  expect_snapshot(x)
  expect_true(inherits(x, "AbstractArray"))
  expect_true(eq_type(x, x, ambiguity = TRUE))

  expect_false(
    eq_type(
      x,
      AbstractArray(
        FloatType(32),
        Shape(c(2, 1))
      ),
      ambiguity = TRUE
    )
  )

  expect_false(
    eq_type(
      x,
      AbstractArray(
        FloatType(64),
        Shape(c(2, 3))
      ),
      ambiguity = TRUE
    )
  )
})

test_that("ConcreteArray", {
  x <- ConcreteArray(
    nv_array(1:6, dtype = "f32", shape = c(2, 3), device = "cpu")
  )
  expect_true(inherits(x, "ConcreteArray"))
  expect_snapshot(x)
})

test_that("from DataType", {
  expect_class(nv_array(1L, "i32"), "AnvilArray")
  expect_class(nv_scalar(1L, "i32"), "AnvilArray")
  expect_class(nv_empty("i32", c(0, 1)), "AnvilArray")
})

test_that("nv_array from nv_array", {
  skip_if(!is_cuda())
  x <- nv_array(1, device = "cuda")
  expect_equal(platform(nv_array(x)), "cuda")
  expect_error(nv_array(x, device = "cpu"))
  expect_error(nv_array(x, shape = c(1, 1)))
  expect_error(nv_array(x, dtype = "f64"))
})

test_that("format", {
  expect_equal(format(nv_array(1:4, shape = c(4, 1))), "AnvilArray(dtype=i32, shape=4x1)")
})

test_that("eq_type and neq_type respect ambiguity argument", {
  # With ambiguity = TRUE, different ambiguity means not equal
  expect_true(
    neq_type(AbstractArray("f32", 1L, TRUE), AbstractArray("f32", 1L, FALSE), ambiguity = TRUE)
  )
  expect_true(
    neq_type(AbstractArray("f32", 1L, FALSE), AbstractArray("f32", 1L, TRUE), ambiguity = TRUE)
  )
  expect_true(
    eq_type(AbstractArray("f32", 1L, TRUE), AbstractArray("f32", 1L, TRUE), ambiguity = TRUE)
  )
  # With ambiguity = FALSE, ambiguity is ignored
  expect_true(
    eq_type(AbstractArray("f32", 1L, TRUE), AbstractArray("f32", 1L, FALSE), ambiguity = FALSE)
  )
  expect_true(
    eq_type(AbstractArray("f32", 1L, FALSE), AbstractArray("f32", 1L, TRUE), ambiguity = FALSE)
  )
})

test_that("== and != operators throw errors for AbstractArray", {
  x <- AbstractArray("f32", 1L, FALSE)
  y <- AbstractArray("f32", 1L, FALSE)
  expect_error(x == y, "Use.*eq_type")
  expect_error(x != y, "Use.*neq_type")
})

test_that("to_abstract", {
  # literal
  expect_equal(to_abstract(TRUE), LiteralArray(TRUE, c(), "bool", FALSE))
  expect_equal(to_abstract(1L), LiteralArray(1L, c(), "i32", TRUE))
  expect_equal(to_abstract(1.0), LiteralArray(1.0, c(), "f32", TRUE))
  # anvil array
  x <- nv_array(1:4, dtype = "f32", shape = c(2, 2))
  expect_equal(to_abstract(x), ConcreteArray(x))
  # graph box
  aval <- GraphValue(AbstractArray("f32", c(2, 2), FALSE))
  x <- GraphBox(aval, local_descriptor())
  expect_equal(to_abstract(x), aval$aval)

  # pure
  x <- nv_scalar(1)
  expect_equal(to_abstract(x, pure = TRUE), AbstractArray("f32", c(), FALSE))
  expect_error(to_abstract(list(1, 2)), "is not an array-like object")
})


test_that("as_shape for c() (i.e., NULL)", {
  expect_equal(as_shape(c()), Shape(integer()))
})

test_that("AbstractArray can be created with any ambiguous dtype", {
  expect_true(ambiguous(AbstractArray("i16", integer(), TRUE)))
})

test_that("nv_abstract creates AbstractArray", {
  expect_equal(
    nv_abstract("f32", c()),
    AbstractArray("f32", Shape(integer()), FALSE)
  )
  expect_equal(
    nv_abstract(as_dtype("i32"), 1:2),
    AbstractArray("i32", Shape(1:2), FALSE)
  )
})

test_that("as_shape for c() (i.e., NULL)", {
  expect_equal(as_shape(c()), Shape(integer()))
})


test_that("to_abstract", {
  # literal
  expect_equal(to_abstract(TRUE), LiteralArray(TRUE, c(), "bool", FALSE))
  expect_equal(to_abstract(1L), LiteralArray(1L, c(), "i32", TRUE))
  expect_equal(to_abstract(1.0), LiteralArray(1.0, c(), "f32", TRUE))
  # anvil array
  x <- nv_array(1:4, dtype = "f32", shape = c(2, 2))
  expect_equal(to_abstract(x), ConcreteArray(x))
  # graph box
  aval <- GraphValue(AbstractArray("f32", c(2, 2), FALSE))
  x <- GraphBox(aval, local_descriptor())
  expect_equal(to_abstract(x), aval$aval)
})

test_that("stablehlo dtype is printed", {
  skip_if(!is_cpu())
  expect_snapshot(nv_array(TRUE))
})

test_that("QuickrDeviceCpu is a classed object", {
  dev <- QuickrDeviceCpu()
  expect_s3_class(dev, "QuickrDeviceCpu")
  expect_equal(format(dev), "QuickrDeviceCpu")
  expect_equal(as.character(dev), "cpu")
})

test_that("PlainDeviceCpu is a classed object", {
  dev <- PlainDeviceCpu()
  expect_s3_class(dev, "PlainDeviceCpu")
  expect_equal(format(dev), "PlainDeviceCpu")
  expect_equal(as.character(dev), "cpu")
})

test_that("device returns QuickrDeviceCpu for quickr arrays", {
  local_backend("quickr")
  x <- nv_array(1)
  dev <- device(x)
  expect_s3_class(dev, "QuickrDeviceCpu")
})

test_that("device returns PlainDeviceCpu for plain arrays", {
  x <- globals$backends[["plain"]]$constructor(1, "f32", 1L, NULL, FALSE)
  dev <- device(x)
  expect_s3_class(dev, "PlainDeviceCpu")
})

test_that("platform returns 'cpu' for quickr backend", {
  local_backend("quickr")
  expect_equal(platform(nv_array(1)), "cpu")
})

test_that("platform returns 'cpu' for plain backend", {
  x <- globals$backends[["plain"]]$constructor(1, "f32", 1L, NULL, FALSE)
  expect_equal(platform(x), "cpu")
})

test_that("backend() returns the backend name", {
  x <- nv_array(1)
  expect_equal(backend(x), "xla")

  local_backend("quickr")
  y <- nv_array(1)
  expect_equal(backend(y), "quickr")
})

test_that("local_backend sets and restores the default backend", {
  old <- getOption("anvil.default_backend")
  local_backend("quickr")
  expect_equal(getOption("anvil.default_backend"), "quickr")
  expect_equal(backend(nv_array(1)), "quickr")
  # auto-restores when scope exits (but we're still in the test scope)
})

test_that("local_backend rejects 'auto'", {
  expect_error(local_backend("auto"), "auto")
})

test_that("nv_array respects backend argument", {
  local_backend("quickr")
  x <- nv_array(1, backend = "xla")
  expect_equal(backend(x), "xla")
})

test_that("nv_array errors when backend specified inside jit", {
  expect_error(
    jit(function() nv_array(1, backend = "xla"))(),
    "must not be specified"
  )
})

test_that("default floating dtype is f32 regardless of backend", {
  expect_equal(dtype(nv_array(1.0)), as_dtype("f32"))
  expect_equal(dtype(nv_scalar(1.0)), as_dtype("f32"))

  local_backend("quickr")
  expect_equal(dtype(nv_array(1.0)), as_dtype("f32"))
  expect_equal(dtype(nv_scalar(1.0)), as_dtype("f32"))
})
