# the main tests are in {pjrt}, hence we keep them minimal here

test_that("nv_serialize and nv_unserialize work for single tensor", {
  x <- nv_tensor(array(rnorm(12), dim = c(3, 4)))
  lst <- list(x = x)
  raw_data <- nv_serialize(lst)
  expect_type(raw_data, "raw")
  reloaded <- nv_unserialize(raw_data)
  expect_equal(lst, reloaded)
})

test_that("nv_write and nv_read works for a single tensor", {
  x <- nv_tensor(array(rnorm(12), dim = c(3, 4)))
  lst <- list(x = x)
  tmp <- tempfile(fileext = ".safetensors")
  nv_write(lst, tmp)
  reloaded <- nv_read(tmp)
  expect_equal(lst, reloaded)
})
