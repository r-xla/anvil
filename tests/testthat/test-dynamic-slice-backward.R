test_that("p_dynamic_slice backward", {
  f <- function(operand) {
    start_idx1 <- nv_scalar(2L, dtype = "i32")
    start_idx2 <- nv_scalar(2L, dtype = "i32")
    sliced <- nvl_dynamic_slice(operand, start_idx1, start_idx2, slice_sizes = c(2L, 2L))
    # Sum to get a scalar output for gradient
    nv_reduce_sum(sliced, dims = c(1L, 2L), drop = TRUE)
  }
  
  g <- jit(gradient(f))
  operand <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
  grads <- g(operand)
  
  # Gradient should be 1 for the sliced elements and 0 elsewhere
  # Slicing at (2, 2) with size (2, 2) means positions [2:3, 2:3]
  # Expected gradient matrix:
  # [0, 0, 0, 0]
  # [0, 1, 1, 0]
  # [0, 1, 1, 0]
  expected <- matrix(c(0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0), nrow = 3, ncol = 4)
  expect_equal(as_array(grads[[1L]]), expected)
})

test_that("p_dynamic_update_slice backward wrt operand", {
  f <- function(operand) {
    update <- nv_tensor(c(99, 88, 77, 66), dtype = "f32", shape = c(2, 2))
    start_idx1 <- nv_scalar(2L, dtype = "i32")
    start_idx2 <- nv_scalar(2L, dtype = "i32")
    updated <- nvl_dynamic_update_slice(operand, update, start_idx1, start_idx2)
    # Sum to get a scalar output for gradient
    nv_reduce_sum(updated, dims = c(1L, 2L), drop = TRUE)
  }
  
  g <- jit(gradient(f))
  operand <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
  grads <- g(operand)
  
  # Gradient for operand: 1 for non-updated elements, 0 for updated elements
  # Updating at (2, 2) with size (2, 2) means positions [2:3, 2:3]
  # Expected gradient matrix:
  # [1, 1, 1, 1]
  # [1, 0, 0, 1]
  # [1, 0, 0, 1]
  expected <- matrix(c(1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1), nrow = 3, ncol = 4)
  expect_equal(as_array(grads[[1L]]), expected)
})

test_that("p_dynamic_update_slice backward wrt update", {
  f <- function(update) {
    operand <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
    start_idx1 <- nv_scalar(2L, dtype = "i32")
    start_idx2 <- nv_scalar(2L, dtype = "i32")
    updated <- nvl_dynamic_update_slice(operand, update, start_idx1, start_idx2)
    # Sum to get a scalar output for gradient
    nv_reduce_sum(updated, dims = c(1L, 2L), drop = TRUE)
  }
  
  g <- jit(gradient(f))
  update <- nv_tensor(c(99, 88, 77, 66), dtype = "f32", shape = c(2, 2))
  grads <- g(update)
  
  # Gradient for update: should be all 1s (since we're summing)
  expected <- matrix(c(1, 1, 1, 1), nrow = 2, ncol = 2)
  expect_equal(as_array(grads[[1L]]), expected)
})

test_that("p_dynamic_slice backward with scalar output", {
  # Test where slice produces a scalar
  f <- function(operand) {
    start_idx1 <- nv_scalar(2L, dtype = "i32")
    start_idx2 <- nv_scalar(3L, dtype = "i32")
    nvl_dynamic_slice(operand, start_idx1, start_idx2, slice_sizes = c(1L, 1L))
  }
  
  g <- jit(gradient(f))
  operand <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
  grads <- g(operand)
  
  # Gradient should be 1 at position (2, 3) and 0 elsewhere
  # Expected gradient matrix:
  # [0, 0, 0, 0]
  # [0, 0, 1, 0]
  # [0, 0, 0, 0]
  expected <- matrix(c(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0), nrow = 3, ncol = 4)
  expect_equal(as_array(grads[[1L]]), expected)
})
