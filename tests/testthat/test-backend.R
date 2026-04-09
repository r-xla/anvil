test_that("default_backend returns 'xla' by default", {
  expect_equal(default_backend(), "xla")
})

test_that("local_backend sets and restores the default backend", {
  skip_if_not_installed("quickr")
  old <- default_backend()
  local_backend("quickr")
  expect_equal(default_backend(), "quickr")
  expect_equal(backend(nv_array(1)), "quickr")
})

test_that("with_backend temporarily changes the backend", {
  skip_if_not_installed("quickr")
  expect_equal(default_backend(), "xla")
  result <- with_backend("quickr", {
    expect_equal(default_backend(), "quickr")
    backend(nv_array(1))
  })
  expect_equal(result, "quickr")
  expect_equal(default_backend(), "xla")
})

test_that("with_backend restores backend on error", {
  skip_if_not_installed("quickr")
  expect_equal(default_backend(), "xla")
  try(with_backend("quickr", stop("test error")), silent = TRUE)
  expect_equal(default_backend(), "xla")
})

test_that("backend() returns the backend name", {
  expect_equal(backend(nv_array(1)), "xla")
})

test_that("backend() returns 'quickr' for quickr arrays", {
  skip_if_not_installed("quickr")
  local_backend("quickr")
  expect_equal(backend(nv_array(1)), "quickr")
})

test_that("nv_empty works with quickr backend", {
  skip_if_not_installed("quickr")
  local_backend("quickr")
  x <- nv_empty("f64", c(0L, 3L))
  expect_equal(backend(x), "quickr")
  expect_equal(dtype(x), as_dtype("f64"))
  expect_equal(shape(x), c(0L, 3L))
})

test_that("nv_empty works with xla backend", {
  x <- nv_empty("f32", c(0L, 3L))
  expect_equal(backend(x), "xla")
  expect_equal(dtype(x), as_dtype("f32"))
  expect_equal(shape(x), c(0L, 3L))
})
