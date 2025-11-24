test_that("p_rnorm", {
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f64", shape_out = c(200L, 300L, 400L))
  }
  g <- jit(f)
  out <- g()
  Z <- as_array(out[[2]])
  expect_false(any(!is.finite(Z)))

  tst <- apply(Z, c(2, 3), \(z) shapiro.test(z)$p.value)
  expect_equal(mean(c(tst) < .05), .05, tolerance = 1e-2)

  # # plot qq-plot of 4 random slices
  # par(mfrow = c(2, 2))
  # for (i in sort(sample(200, 4))) {
  #   qqnorm(Z[i, , ], col = "grey", main = sprintf("Q-Q Plot of Z[%d, , ]", i))
  #   qqline(Z[i, , ], lwd = 2)
  # }
})
