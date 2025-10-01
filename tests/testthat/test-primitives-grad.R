test_that("nvl_add (scalar)", {
  expect_grad_binary(nvl_add, \(x, y) 1, \(x, y) 1, 1.2, -0.7)
})

test_that("nvl_sub (scalar)", {
  expect_grad_binary(nvl_sub, \(x, y) 1, \(x, y) -1, 1.2, -0.7)
})

test_that("nvl_mul", {
  expect_grad_binary(nvl_mul, \(x, y) y, \(x, y) x, 1.2, -0.7)
})

test_that("nvl_neg", {
  expect_grad_unary(nvl_neg, `-`, 1)
})
