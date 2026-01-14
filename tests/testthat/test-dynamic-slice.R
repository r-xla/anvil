test_that("p_dynamic_slice basic", {
  f <- function() {
    operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
    start_idx1 <- nv_scalar(2L, dtype = "i32")
    start_idx2 <- nv_scalar(2L, dtype = "i32")
    nvl_dynamic_slice(operand, start_idx1, start_idx2, slice_sizes = c(2L, 2L))
  }
  g <- jit(f)
  out <- g()
  # Starting at (2, 2) (1-based), slice size (2, 2)
  # Should get elements at positions [2:3, 2:3]
  # Matrix is filled column-wise: 1:12
  # [1, 4, 7, 10]
  # [2, 5, 8, 11]
  # [3, 6, 9, 12]
  # Slice [2:3, 2:3] gives us:
  # [5, 8]
  # [6, 9]
  expected <- matrix(c(5L, 6L, 8L, 9L), nrow = 2, ncol = 2)
  expect_equal(as_array(out), expected)
})

test_that("p_dynamic_update_slice basic", {
  f <- function() {
    operand <- nv_tensor(rep(0L, 12), dtype = "i32", shape = c(3, 4))
    update <- nv_tensor(c(99L, 88L, 77L, 66L), dtype = "i32", shape = c(2, 2))
    start_idx1 <- nv_scalar(2L, dtype = "i32")
    start_idx2 <- nv_scalar(2L, dtype = "i32")
    nvl_dynamic_update_slice(operand, update, start_idx1, start_idx2)
  }
  g <- jit(f)
  out <- g()
  # Starting at (2, 2) (1-based), update with (2, 2) tensor
  # Should update elements at positions [2:3, 2:3]
  # Expected:
  # [0, 0,  0,  0]
  # [0, 99, 77, 0]
  # [0, 88, 66, 0]
  expected <- matrix(c(0L, 0L, 0L, 0L, 99L, 88L, 0L, 77L, 66L, 0L, 0L, 0L), nrow = 3, ncol = 4)
  expect_equal(as_array(out), expected)
})

test_that("p_dynamic_slice with nv_* API", {
  f <- function() {
    operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
    start_idx1 <- nv_scalar(1L, dtype = "i32")
    start_idx2 <- nv_scalar(3L, dtype = "i32")
    nv_dynamic_slice(operand, start_idx1, start_idx2, slice_sizes = c(2L, 2L))
  }
  g <- jit(f)
  out <- g()
  # Starting at (1, 3) (1-based), slice size (2, 2)
  # Matrix is:
  # [1, 4, 7, 10]
  # [2, 5, 8, 11]
  # [3, 6, 9, 12]
  # Slice [1:2, 3:4] gives us:
  # [7, 10]
  # [8, 11]
  expected <- matrix(c(7L, 8L, 10L, 11L), nrow = 2, ncol = 2)
  expect_equal(as_array(out), expected)
})

test_that("p_dynamic_update_slice with nv_* API", {
  f <- function() {
    operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
    update <- nv_tensor(c(99L, 88L, 77L, 66L), dtype = "i32", shape = c(2, 2))
    start_idx1 <- nv_scalar(1L, dtype = "i32")
    start_idx2 <- nv_scalar(1L, dtype = "i32")
    nv_dynamic_update_slice(operand, update, start_idx1, start_idx2)
  }
  g <- jit(f)
  out <- g()
  # Starting at (1, 1) (1-based), update with (2, 2) tensor
  # Original matrix:
  # [1, 4, 7, 10]
  # [2, 5, 8, 11]
  # [3, 6, 9, 12]
  # After update at [1:2, 1:2]:
  # [99, 77, 7, 10]
  # [88, 66, 8, 11]
  # [3,  6,  9, 12]
  expected <- matrix(c(99L, 88L, 3L, 77L, 66L, 6L, 7L, 8L, 9L, 10L, 11L, 12L), nrow = 3, ncol = 4)
  expect_equal(as_array(out), expected)
})
