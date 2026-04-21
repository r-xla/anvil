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
  expect_equal(platform(x), "cuda")
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

test_that("nv_aval creates AbstractArray", {
  expect_equal(
    nv_aval("f32", c()),
    AbstractArray("f32", Shape(integer()), FALSE)
  )
  expect_equal(
    nv_aval(as_dtype("i32"), 1:2),
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

test_that("quickr_device is a classed object", {
  skip_if_quickr()
  dev <- quickr_device("cpu")
  expect_s3_class(dev, "QuickrDevice")
  expect_equal(format(dev), "QuickrDevice(cpu)")
  expect_equal(as.character(dev), "cpu")
})

test_that("PlainDeviceCpu is a classed object", {
  dev <- PlainDeviceCpu()
  expect_s3_class(dev, "PlainDeviceCpu")
  expect_equal(format(dev), "PlainDeviceCpu")
  expect_equal(as.character(dev), "cpu")
})

test_that("device returns QuickrDevice for quickr arrays", {
  skip_if_quickr()
  local_backend("quickr")
  x <- nv_array(1)
  dev <- device(x)
  expect_s3_class(dev, "QuickrDevice")
})

test_that("device returns PlainDeviceCpu for plain arrays", {
  x <- globals$backends[["plain"]]$new_data(1, "f32", 1L, NULL, FALSE)
  dev <- device(x)
  expect_s3_class(dev, "PlainDeviceCpu")
})

test_that("platform returns 'cpu' for quickr backend", {
  skip_if_quickr()
  local_backend("quickr")
  expect_equal(platform(nv_array(1)), "cpu")
})

test_that("platform returns 'cpu' for plain backend", {
  x <- globals$backends[["plain"]]$new_data(1, "f32", 1L, NULL, FALSE)
  expect_equal(platform(x), "cpu")
})

test_that("nv_array respects backend argument", {
  skip_if_quickr()
  local_backend("quickr")
  x <- nv_array(1, backend = "xla")
  expect_equal(backend(x), "xla")
})

test_that("nv_array infers backend from device object", {
  skip_if_quickr()
  local_backend("quickr")
  x <- nv_array(1, device = pjrt::pjrt_device("cpu"))
  expect_equal(backend(x), "xla")
  expect_equal(device(x), nv_device("cpu", "xla"))
})

test_that("nv_array errors when backend specified inside jit", {
  expect_error(
    jit(function() nv_array(1, backend = "xla"))(),
    "must not be specified"
  )
})

test_that("default floating dtype is f32 for xla", {
  expect_equal(dtype(nv_array(1.0)), as_dtype("f32"))
  expect_equal(dtype(nv_scalar(1.0)), as_dtype("f32"))
})

test_that("default floating dtype is f64 for quickr", {
  skip_if_quickr()
  local_backend("quickr")
  expect_equal(dtype(nv_array(1.0)), as_dtype("f64"))
  expect_equal(dtype(nv_scalar(1.0)), as_dtype("f64"))
})

test_that("nv_array_like inherits dtype, shape, ambiguous, device, backend from like", {
  like <- nv_array(c(1L, 2L, 3L), dtype = "i16", ambiguous = TRUE)
  out <- nv_array_like(like, c(7L, 8L, 9L))
  expect_equal(dtype(out), dtype(like))
  expect_equal(shape(out), shape(like))
  expect_equal(ambiguous(out), ambiguous(like))
  expect_equal(backend(out), backend(like))
  expect_equal(as.integer(as_array(out)), c(7L, 8L, 9L))
})

test_that("nv_array_like respects explicit overrides", {
  like <- nv_array(c(1L, 2L, 3L), dtype = "i16", ambiguous = TRUE)
  out <- nv_array_like(like, c(1L, 2L, 3L, 4L), dtype = "i32", ambiguous = FALSE, shape = 4L)
  expect_equal(dtype(out), as_dtype("i32"))
  expect_equal(shape(out), 4L)
  expect_false(ambiguous(out))
})

test_that("nv_scalar_like inherits dtype, ambiguous, device, backend from like", {
  like <- nv_scalar(1L, dtype = "i16", ambiguous = TRUE)
  out <- nv_scalar_like(like, 7L)
  expect_equal(dtype(out), dtype(like))
  expect_equal(shape(out), integer())
  expect_equal(ambiguous(out), ambiguous(like))
  expect_equal(backend(out), backend(like))
  expect_equal(as.integer(as_array(out)), 7L)
})

describe("as_anvil_array", {
  it("passes AnvilArrays through unchanged", {
    x <- nv_array(1:3)
    expect_identical(as_anvil_array(x), x)
  })

  it("converts scalar R literals into scalar AnvilArrays", {
    out <- as_anvil_array(1L)
    expect_s3_class(out, "AnvilArray")
    expect_equal(shape(out), integer())
    expect_true(ambiguous(out))
  })

  it("converts R arrays into AnvilArrays preserving shape", {
    out <- as_anvil_array(array(1:6, c(2, 3)))
    expect_s3_class(out, "AnvilArray")
    expect_equal(shape(out), c(2L, 3L))
  })

  it("places R literals on the requested device", {
    dev <- nv_device("cpu:1", "xla")
    expect_equal(device(as_anvil_array(1L, device = dev)), dev)
  })

  it("errors if an AnvilArray is on a different device than requested", {
    dev0 <- nv_device("cpu:0", "xla")
    dev1 <- nv_device("cpu:1", "xla")
    x <- nv_array(1:3, device = dev0)
    expect_error(
      as_anvil_array(x, device = dev1),
      "unexpected device"
    )
  })

  it("rejects non-arrayish inputs", {
    expect_error(as_anvil_array("foo"), "Expected arrayish")
    expect_error(as_anvil_array(list()), "Expected arrayish")
  })

  it("passes traced boxes through unchanged under jit()", {
    f <- jit(function(x) {
      y <- as_anvil_array(x)
      y + 1
    })
    out <- f(nv_array(1:3))
    expect_equal(as_array(out), array(2:4, dim = 3L))
  })

  it("handles R literals under jit()", {
    f <- jit(function() as_anvil_array(1L) + 1L)
    out <- f()
    expect_s3_class(out, "AnvilArray")
    expect_equal(as.integer(as_array(out)), 2L)
  })
})

describe("as_anvil_arrays", {
  it("places R literals on the first concrete input's device", {
    dev <- nv_device("cpu:1", "xla")
    x <- nv_array(1:3, device = dev)
    out <- as_anvil_arrays(x, 1L)
    expect_equal(device(out[[1L]]), dev)
    expect_equal(device(out[[2L]]), dev)
  })

  it("uses the default device when no concrete input is present", {
    out <- as_anvil_arrays(1L, 2L)
    expect_equal(device(out[[1L]]), default_device())
    expect_equal(device(out[[2L]]), default_device())
  })

  it("errors when concrete inputs live on different devices", {
    dev0 <- nv_device("cpu:0", "xla")
    dev1 <- nv_device("cpu:1", "xla")
    x <- nv_array(1:3, device = dev0)
    y <- nv_array(1:3, device = dev1)
    expect_error(
      as_anvil_arrays(x, y),
      "multiple devices"
    )
  })

  it("errors when concrete inputs come from different backends", {
    skip_if_quickr()
    dev_xla <- nv_device("cpu", "xla")
    dev_quickr <- nv_device("cpu", "quickr")
    x <- nv_array(1:3, device = dev_xla)
    y <- nv_array(1:3, device = dev_quickr)
    expect_error(
      as_anvil_arrays(x, y),
      "multiple backends"
    )
  })

  it("passes concrete inputs on the same device through unchanged", {
    x <- nv_array(1:3)
    y <- nv_array(4:6)
    out <- as_anvil_arrays(x, y)
    expect_identical(out[[1L]], x)
    expect_identical(out[[2L]], y)
  })

  it("canonicalizes mixed traced and literal inputs under jit()", {
    f <- jit(function(x) {
      args <- as_anvil_arrays(x, 1L)
      args[[1L]] + args[[2L]]
    })
    out <- f(nv_array(1:3))
    expect_equal(as.integer(as_array(out)), 2:4)
  })

  it("canonicalizes multiple traced inputs under jit()", {
    f <- jit(function(x, y) {
      args <- as_anvil_arrays(x, y)
      args[[1L]] + args[[2L]]
    })
    out <- f(nv_array(1:3), nv_array(4:6))
    expect_equal(as.integer(as_array(out)), c(5L, 7L, 9L))
  })
})
