test_that("nvl_add", {
  expect_jit_binary(nvl_add, `+`, 1.2, -0.7)
})

test_that("nvl_sub", {
  expect_jit_binary(nvl_sub, `-`, 1.2, -0.7)
})

test_that("nvl_mul", {
  expect_jit_binary(nvl_mul, `*`, 1.2, -0.7)
})

test_that("nvl_neg", {
  expect_jit_unary(nvl_neg, \(x) -x, 1.7)
})
