test_that("p_runif", {
  f <- function() {
    nv_runif(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape_out = c(2, 3), lower = -1, upper = 1)
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 5L))
  expect_equal(
    as_array(out[[2]]),
    array(c(-0.9798, 0.0015, 0.8532, 0.7442, -0.2211, 0.9746), c(2, 3)),
    tolerance = 1e-4
  )
})
