describe("nv_dynamic_slice", {
  it("extracts a slice at dynamic position", {
    f <- function() {
      operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
      start_idx1 <- nv_scalar(1L, dtype = "i32")
      start_idx2 <- nv_scalar(3L, dtype = "i32")
      nv_dynamic_slice(operand, start_idx1, start_idx2, slice_sizes = c(2L, 2L))
    }
    g <- jit(f)
    out <- g()
    expected <- matrix(c(7L, 8L, 10L, 11L), nrow = 2, ncol = 2)
    expect_equal(as_array(out), expected)
  })

  it("clamps negative indices to 1", {
    f <- function() {
      operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
      start_idx1 <- nv_scalar(-1L, dtype = "i32")
      start_idx2 <- nv_scalar(1L, dtype = "i32")
      nv_dynamic_slice(operand, start_idx1, start_idx2, slice_sizes = c(2L, 2L))
    }
    g <- jit(f)
    out <- g()
    # Negative index is clamped to 1, so slices from [1, 1]
    expected <- matrix(c(1L, 2L, 4L, 5L), nrow = 2, ncol = 2)
    expect_equal(as_array(out), expected)
  })

  it("clamps indices to prevent out of bounds", {
    f <- function() {
      operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
      start_idx1 <- nv_scalar(3L, dtype = "i32")
      start_idx2 <- nv_scalar(4L, dtype = "i32")
      # Would go out of bounds, clamped to [2, 3] (3-2+1=2, 4-2+1=3)
      nv_dynamic_slice(operand, start_idx1, start_idx2, slice_sizes = c(2L, 2L))
    }
    g <- jit(f)
    out <- g()
    expected <- matrix(c(8L, 9L, 11L, 12L), nrow = 2, ncol = 2)
    expect_equal(as_array(out), expected)
  })
})

describe("nv_dynamic_update_slice", {
  it("updates a slice at dynamic position", {
    f <- function() {
      operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
      update <- nv_tensor(c(99L, 88L, 77L, 66L), dtype = "i32", shape = c(2, 2))
      start_idx1 <- nv_scalar(1L, dtype = "i32")
      start_idx2 <- nv_scalar(1L, dtype = "i32")
      nv_dynamic_update_slice(operand, update, start_idx1, start_idx2)
    }
    g <- jit(f)
    out <- g()
    expected <- matrix(c(99L, 88L, 3L, 77L, 66L, 6L, 7L, 8L, 9L, 10L, 11L, 12L), nrow = 3, ncol = 4)
    expect_equal(as_array(out), expected)
  })

  it("clamps negative indices to 1", {
    f <- function() {
      operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
      update <- nv_tensor(c(99L, 88L, 77L, 66L), dtype = "i32", shape = c(2, 2))
      start_idx1 <- nv_scalar(-1L, dtype = "i32")
      start_idx2 <- nv_scalar(1L, dtype = "i32")
      nv_dynamic_update_slice(operand, update, start_idx1, start_idx2)
    }
    g <- jit(f)
    out <- g()
    # Negative index is clamped to 1
    expected <- matrix(c(99L, 88L, 3L, 77L, 66L, 6L, 7L, 8L, 9L, 10L, 11L, 12L), nrow = 3, ncol = 4)
    expect_equal(as_array(out), expected)
  })

  it("clamps indices to prevent out of bounds", {
    f <- function() {
      operand <- nv_tensor(1:12, dtype = "i32", shape = c(3, 4))
      update <- nv_tensor(c(99L, 88L, 77L, 66L), dtype = "i32", shape = c(2, 2))
      start_idx1 <- nv_scalar(3L, dtype = "i32")
      start_idx2 <- nv_scalar(4L, dtype = "i32")
      # Would go out of bounds, clamped to [2, 3] (3-2+1=2, 4-2+1=3)
      nv_dynamic_update_slice(operand, update, start_idx1, start_idx2)
    }
    g <- jit(f)
    out <- g()
    expected <- matrix(c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 99L, 88L, 10L, 77L, 66L), nrow = 3, ncol = 4)
    expect_equal(as_array(out), expected)
  })
})

describe("nv_get_elt", {
  it("extracts single element from 2D tensor", {
    f <- function() {
      x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
      idx1 <- nv_scalar(2L, dtype = "i32")
      idx2 <- nv_scalar(3L, dtype = "i32")
      nv_get_elt(x, idx1, idx2)
    }
    g <- jit(f)
    result <- g()
    # Element at (2, 3) in 1-based indexing
    # Matrix filled column-wise: [1,4,7,10; 2,5,8,11; 3,6,9,12]
    # (2, 3) -> 8
    expect_equal(as_array(result), 8)
    expect_equal(dim(as_array(result)), NULL) # scalar
  })

  it("works with 1D tensor", {
    f <- function() {
      x <- nv_tensor(c(10, 20, 30, 40, 50), dtype = "f32")
      idx <- nv_scalar(3L, dtype = "i32")
      nv_get_elt(x, idx)
    }
    g <- jit(f)
    result <- g()
    expect_equal(as_array(result), 30)
  })

  it("fails with non-scalar indices", {
    expect_error(
      {
        f <- function() {
          x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
          idx1 <- nv_tensor(c(1L, 2L), dtype = "i32") # Not a scalar!
          idx2 <- nv_scalar(1L, dtype = "i32")
          nv_get_elt(x, idx1, idx2)
        }
        g <- jit(f)
        g()
      },
      "0-dimensional"
    )
  })

  it("fails with wrong number of indices", {
    expect_error(
      {
        f <- function() {
          x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
          idx <- nv_scalar(1L, dtype = "i32")
          nv_get_elt(x, idx) # Only 1 index for 2D tensor
        }
        g <- jit(f)
        g()
      },
      "Number of indices must match tensor rank"
    )
  })
})

describe("nv_set_elt", {
  it("updates single element in 2D tensor", {
    f <- function() {
      x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
      idx1 <- nv_scalar(2L, dtype = "i32")
      idx2 <- nv_scalar(3L, dtype = "i32")
      value <- nv_scalar(99, dtype = "f32")
      nv_set_elt(x, idx1, idx2, value = value)
    }
    g <- jit(f)
    result <- g()
    expected <- matrix(1:12, nrow = 3, ncol = 4)
    expected[2, 3] <- 99
    expect_equal(as_array(result), expected)
  })

  it("works with 1D tensor", {
    f <- function() {
      x <- nv_tensor(c(10, 20, 30, 40, 50), dtype = "f32")
      idx <- nv_scalar(2L, dtype = "i32")
      value <- nv_scalar(999, dtype = "f32")
      nv_set_elt(x, idx, value = value)
    }
    g <- jit(f)
    result <- g()
    expect_equal(as_array(result), array(c(10, 999, 30, 40, 50), dim = 5))
  })

  it("fails with non-scalar value", {
    expect_error(
      {
        f <- function() {
          x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
          idx1 <- nv_scalar(1L, dtype = "i32")
          idx2 <- nv_scalar(1L, dtype = "i32")
          value <- nv_tensor(c(1, 2), dtype = "f32") # Not a scalar!
          nv_set_elt(x, idx1, idx2, value = value)
        }
        g <- jit(f)
        g()
      },
      "0-dimensional"
    )
  })
})
