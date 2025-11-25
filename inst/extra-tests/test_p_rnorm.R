test_that("p_rnorm", {
  # basic test
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape_out = c(2, 3))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 6L))
  expect_equal(
    as_array(out[[2]]),
    array(c(1.9399, -2.3290, -0.0936, 1.1724, 0.0217, -0.3899), c(2, 3)),
    tolerance = 1e-4
  )

  # test with uneven total number of RVs
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape_out = c(3, 3))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 8L))
  expect_equal(shape(out[[2]]), c(3L, 3L))

  # check normality
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f64", shape_out = c(200L, 300L, 400L))
  }
  g <- jit(f)
  out <- g()
  Z <- as_array(out[[2]])
  expect_false(any(!is.finite(Z)))

  tst <- apply(Z, c(2, 3), \(z) shapiro.test(z)$p.value)
  expect_equal(round(mean(c(tst) < .05), 2), .05)

  # # plot qq-plot of 4 random slices
  # par(mfrow = c(2, 2))
  # for (i in sort(sample(200, 4))) {
  #   qqnorm(Z[i, , ], col = "grey", main = sprintf("Q-Q Plot of Z[%d, , ]", i))
  #   qqline(Z[i, , ], lwd = 2)
  # }

  f <- function() {
    nv_rnorm(
      nv_tensor(c(3, 83), dtype = "ui64"),
      dtype = "f64",
      shape_out = c(10L, 10L, 10L, 10L, 10L),
      mu = 10,
      sigma = 9
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(round(mean(as_array(out[[2]])), 1), 10)
  expect_equal(round(sd(as_array(out[[2]])), 1), 9)
})
