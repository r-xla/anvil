skip_if_not_installed("safetensors")

test_that("nv_serialize and nv_unserialize work for single tensor", {
  # the main tests are in {pjrt}
  x <- nv_tensor(array(rnorm(12), dim = c(3, 4)))
  raw_data <- nv_serialize(list(x = x))
  expect_type(raw_data, "raw")
  reloaded <- nv_unserialize(raw_data)
  expect_true(inherits(reloaded$x, "AnvilTensor"))
  expect_equal(x, reloaded$x)
})

test_that("nv_write and nv_read works for a single tensor", {
  x <- nv_tensor(array(rnorm(12), dim = c(3, 4)))
  tmp <- tempfile(fileext = ".safetensors")
  nv_write(list(x = x), tmp)
  reloaded <- nv_read(tmp)
  expect_equal(x, reloaded$x)
})
