test_that("nv_device dispatches to the backend-specific constructor", {
  skip_if_not_installed("quickr")
  dev <- nv_device("cpu", backend = "quickr")
  expect_s3_class(dev, "QuickrDevice")
  expect_equal(backend(dev), "quickr")
})

test_that("nv_device returns PJRTDevice for xla backend", {
  skip_if(!pjrt::plugin_is_downloaded())
  dev <- nv_device("cpu", backend = "xla")
  expect_s3_class(dev, "PJRTDevice")
  expect_equal(backend(dev), "xla")
})

test_that("nv_device errors on the plain backend", {
  expect_error(nv_device("cpu", backend = "plain"), "plain")
})
