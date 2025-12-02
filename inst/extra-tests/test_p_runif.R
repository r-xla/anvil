test_that("p_runif", {
  f <- function() {
    nv_runif(
      nv_tensor(c(1, 2), dtype = "ui64"),
      dtype = "f32",
      shape_out = c(10, 20, 30, 40, 50),
      lower = -1,
      upper = 1
    )
  }
  g <- jit(f)
  out <- g()

  expect_false(any(as_array(out[[2]]) == -1))
  expect_false(any(as_array(out[[2]]) == 1))

  expect_equal(c(as_array(out[[1]])), c(1L, 6000002L))
  expect_equal(
    shape(out[[2]]),
    c(10, 20, 30, 40, 50)
  )
})
