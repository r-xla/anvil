test_that("generate state", {
  f <- function() {
    nv_runif(
      initial_state = nv_rng_state(1),
      dtype = "f64",
      shape_out = c(10, 5)
    )
  }
  g <- jit(f)
  out1 <- g()

  f <- function() {
    nv_runif(
      initial_state = nv_rng_state(1),
      dtype = "f64",
      shape_out = c(10, 5)
    )
  }
  g <- jit(f)
  out2 <- g()

  # check resulting states
  expect_equal(as_array(out1[[1]]), as_array(out2[[1]]))
  # check random variables
  expect_equal(as_array(out1[[2]]), as_array(out2[[2]]))
})
