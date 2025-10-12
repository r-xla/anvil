verify_grad_uni_scalar <- function(.f, .g) {
  lhs <- rnorm(1)

  operand_anvil <- nv_scalar(lhs)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  operand_torch <- torch::torch_tensor(lhs, requires_grad = TRUE)

  grads_anvil <- jit(gradient(.f))(operand_anvil)
  out <- .g(operand_torch$squeeze())
  out$backward(retrain_graph = TRUE)

  expect_equal(
    as_array(grads_anvil[[1L]]),
    torch::as_array(operand_torch$grad)
  )
}

verify_grad_uni_tensor <- function(.f, .g) {
  operand <- array(rnorm(20), c(1, 4, 5))

  operand_anvil <- nv_tensor(operand)

  operand_torch <- torch::torch_tensor(operand, requires_grad = TRUE)

  .f_anvil <- \(operand) {
    nv_reduce_sum(.f(operand), dims = 1:3)
  }
  .g_torch <- \(operand) {
    torch::torch_sum(.g(operand), dim = 1:3)
  }

  grads_anvil <- jit(gradient(.f_anvil))(operand_anvil)
  .g_torch(operand_torch)$backward()

  expect_equal(
    as_array(grads_anvil[[1L]]),
    torch::as_array(operand_torch$grad)
  )
}

verify_grad_biv_scalar <- function(.f, .g) {
  lhs <- rnorm(1)
  rhs <- rnorm(1)

  lhs_anvil <- nv_scalar(lhs)
  rhs_anvil <- nv_scalar(rhs)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  lhs_torch <- torch::torch_tensor(lhs, requires_grad = TRUE)

  rhs_torch <- torch::torch_tensor(rhs, requires_grad = TRUE)

  grads_anvil <- jit(gradient(.f))(lhs_anvil, rhs_anvil)
  out <- .g(lhs_torch$squeeze(), rhs_torch$squeeze())
  out$backward(retrain_graph = TRUE)

  expect_equal(
    as_array(grads_anvil[[1L]]),
    torch::as_array(lhs_torch$grad)
  )

  expect_equal(
    as_array(grads_anvil[[2L]]),
    torch::as_array(rhs_torch$grad)
  )
}

verify_grad_biv_tensor <- function(.f, .g) {
  lhs <- array(rnorm(20), c(1, 4, 5))
  rhs <- array(rnorm(20), c(1, 4, 5))

  lhs_anvil <- nv_tensor(lhs)
  rhs_anvil <- nv_tensor(rhs)

  lhs_torch <- torch::torch_tensor(lhs, requires_grad = TRUE)
  rhs_torch <- torch::torch_tensor(rhs, requires_grad = TRUE)

  .f_anvil <- \(lhs, rhs) {
    nv_reduce_sum(.f(lhs, rhs), dims = 1:3)
  }
  .g_torch <- \(lhs, rhs) {
    torch::torch_sum(.g(lhs, rhs), dim = 1:3)
  }

  grads_anvil <- jit(gradient(.f_anvil))(lhs_anvil, rhs_anvil)
  .g_torch(lhs_torch, rhs_torch)$backward()

  expect_equal(
    as_array(grads_anvil[[1L]]),
    torch::as_array(lhs_torch$grad)
  )

  expect_equal(
    as_array(grads_anvil[[2L]]),
    torch::as_array(rhs_torch$grad)
  )
}

verify_grad_biv <- function(f, g) {
  verify_grad_biv_scalar(f, g)
  verify_grad_biv_tensor(f, g)
}

verify_grad_uni <- function(f, g) {
  verify_grad_uni_scalar(f, g)
  verify_grad_uni_tensor(f, g)
}

test_that("add", {
  verify_grad_biv(nvl_add, torch::torch_add)
})

test_that("sub", {
  verify_grad_biv(nvl_sub, torch::torch_sub)
})

test_that("mul", {
  verify_grad_biv(nvl_mul, torch::torch_mul)
})

test_that("neg", {
  verify_grad_uni(nvl_neg, torch::torch_neg)
})

test_that("div", {
  verify_grad_biv(nvl_div, torch::torch_div)
})
