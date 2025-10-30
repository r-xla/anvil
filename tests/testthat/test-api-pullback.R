test_that("nv_mean", {
  f <- function(y, alpha) {
    mean(y - alpha)
  }

  alpha <- nv_scalar(0.5)
  y <- nv_tensor(1:10, "f32")
  out <- jit(gradient(f, wrt = "alpha"))(y, alpha)
  expect_equal(shape(out[[1L]]), integer())
})
