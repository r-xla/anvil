describe("[", {
  # main tests are in test-api-subset.R
  it("extracts single element", {
    expect_jit_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(1L, dtype = "i32")
        idx2 <- nv_scalar(1L, dtype = "i32")
        x[idx1, idx2]
      },
      nv_scalar(1, ambiguous = FALSE)
    )
  })

  it("can use variables as indices", {
    expect_jit_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(2L, dtype = "i32")
        idx2 <- nv_scalar(3L, dtype = "i32")
        x[idx1, idx2]
      },
      # Scalar array indices drop dimensions, so result is a scalar
      nv_scalar(8, ambiguous = FALSE)
    )
  })
})

describe("!", {
  it("negates boolean array", {
    expect_jit_equal(
      {
        x <- nv_array(c(TRUE, FALSE, TRUE), dtype = "bool")
        !x
      },
      nv_array(c(FALSE, TRUE, FALSE), dtype = "bool")
    )
  })
})

describe("log2", {
  it("computes base-2 logarithm", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 4, 8))
        log2(x)
      },
      nv_array(log2(c(1, 2, 4, 8))),
      tolerance = 1e-6
    )
  })
})

describe("log10", {
  it("computes base-10 logarithm", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 10, 100, 1000))
        log10(x)
      },
      nv_array(log10(c(1, 10, 100, 1000))),
      tolerance = 1e-6
    )
  })
})

describe("expm1", {
  it("computes exp(x) - 1", {
    expect_jit_equal(
      {
        x <- nv_array(c(0, 0.001, 1))
        expm1(x)
      },
      nv_array(expm1(c(0, 0.001, 1))),
      tolerance = 1e-6
    )
  })
})

describe("log1p", {
  it("computes log(1 + x)", {
    expect_jit_equal(
      {
        x <- nv_array(c(0, 0.001, 1))
        log1p(x)
      },
      nv_array(log1p(c(0, 0.001, 1))),
      tolerance = 1e-6
    )
  })
})

describe("[<-", {
  # main tests are in test-api-subset.R
  it("updates single element", {
    expect_jit_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(3L, dtype = "i32")
        idx2 <- nv_scalar(4L, dtype = "i32")
        value <- nv_scalar(-1, dtype = "f32")
        x[3L, 4L] <- value
        x
      },
      nv_array(c(1:11, -1L), dtype = "f32", shape = c(3, 4))
    )
  })

  it("can use variables as indices (NSE)", {
    expect_jit_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(1L, dtype = "i32")
        idx2 <- nv_scalar(2L, dtype = "i32")
        value <- nv_scalar(99, dtype = "f32")
        x[idx1, idx2] <- value
        x
      },
      nv_array(c(1:3, 99, 5:12), dtype = "f32", shape = c(3, 4))
    )
  })
})
