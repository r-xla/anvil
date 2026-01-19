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

describe("c", {
  it("concatenates scalars", {
    expect_equal(jit_eval(c(nv_scalar(1L), nv_scalar(2L))), nv_tensor(1:2))
  })
  it("can concatenate scalars and vectors", {
    expect_equal(jit_eval(c(nv_scalar(1L), 2L)), nv_tensor(1:3))
  })
  it("can concatenate literals", {
    expect_equal(nv_tensor(1:2), jit_eval(c(nv_scalar(1L), 2L)))
    expect_equal(nv_tensor(1:2), jit_eval(c(1L, nv_scalar(2L))))
  })
  it("fails with > 1D tensors", {
    expect_snapshot(
      jit_eval(c(nv_tensor(1:2), nv_tensor(3:4)))
    )
  })
})
