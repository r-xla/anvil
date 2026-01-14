describe("[", {
  it("extracts single element", {
    f <- function() {
      x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
      idx1 <- nv_scalar(1L, dtype = "i32")
      idx2 <- nv_scalar(1L, dtype = "i32")
      x[idx1, idx2]
    }
    g <- jit(f)
    result <- g()
    expect_equal(as_array(result), 1)
  })
})

describe("[<-", {
  it("updates single element", {
    f <- function() {
      x <- nv_tensor(1:12, dtype = "f32", shape = c(3, 4))
      idx1 <- nv_scalar(3L, dtype = "i32")
      idx2 <- nv_scalar(4L, dtype = "i32")
      value <- nv_scalar(-1, dtype = "f32")
      x[idx1, idx2] <- value
      x
    }
    g <- jit(f)
    result <- g()
    expected <- matrix(1:12, nrow = 3, ncol = 4)
    expected[3, 4] <- -1
    expect_equal(as_array(result), expected)
  })
})
