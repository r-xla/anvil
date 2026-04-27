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

describe("length", {
  it("works with 1D array", {
    expect_equal(length(nv_array(1:3)), 3L)
  })
  it("works with scalar", {
    expect_equal(length(nv_scalar(0L)), 1L)
  })
  it("works with 2D array", {
    expect_equal(length(nv_array(1L, shape = c(2L, 3L))), 6L)
  })
})

describe("nrow", {
  it("works with >= 2D array", {
    expect_equal(nrow(nv_array(1:6, shape = c(3L, 2L))), 3L)
  })
  it("returns NULL for < 2D array", {
    expect_equal(nrow(nv_array(1:2)), 2L)
  })
})

describe("ncol", {
  it("works with >= 2D array", {
    expect_equal(ncol(nv_array(1:6, shape = c(3L, 2L))), 2L)
  })
  it("returns NULL for < 2D array", {
    expect_true(is.na(ncol(nv_array(1L))))
  })
})

describe("dim", {
  it("returns shape", {
    x <- nv_array(array(1:24, dim = c(2, 3, 4)))
    expect_equal(dim(x), shape(x))
  })
})

describe("rbind", {
  it("stacks two 1-D vectors as rows (eager)", {
    x <- nv_array(c(1, 2, 3))
    y <- nv_array(c(4, 5, 6))
    expect_equal(
      rbind(x, y),
      nv_array(rbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("stacks two 1-D vectors as rows (jit)", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3))
        y <- nv_array(c(4, 5, 6))
        rbind(x, y)
      },
      nv_array(rbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("stacks two matrices vertically", {
    a <- matrix(1:6, nrow = 2)
    b <- matrix(7:12, nrow = 2)
    expect_jit_equal(
      {
        x <- nv_array(a)
        y <- nv_array(b)
        rbind(x, y)
      },
      nv_array(rbind(a, b))
    )
  })

  it("treats 1-D operand as a row when mixed with a matrix", {
    a <- matrix(1:6, nrow = 2)
    v <- c(7, 8, 9)
    expect_jit_equal(
      {
        x <- nv_array(a)
        y <- nv_array(v)
        rbind(x, y)
      },
      nv_array(rbind(a, v))
    )
  })

  it("accepts more than two arguments", {
    expect_jit_equal(
      {
        rbind(nv_array(c(1, 2)), nv_array(c(3, 4)), nv_array(c(5, 6)))
      },
      nv_array(rbind(c(1, 2), c(3, 4), c(5, 6)))
    )
  })

  it("promotes operands to a common dtype", {
    x <- nv_array(c(1L, 2L, 3L))
    y <- nv_array(c(4, 5, 6))
    out <- rbind(x, y)
    expect_equal(dtype(out), dtype(y))
  })

  it("errors when number of columns mismatch", {
    expect_error(rbind(nv_array(c(1, 2)), nv_array(c(3, 4, 5))))
  })

  it("nv_rbind matches rbind", {
    x <- nv_array(c(1, 2, 3))
    y <- nv_array(c(4, 5, 6))
    expect_equal(nv_rbind(x, y), rbind(x, y))
  })
})

describe("cbind", {
  it("stacks two 1-D vectors as columns (eager)", {
    x <- nv_array(c(1, 2, 3))
    y <- nv_array(c(4, 5, 6))
    expect_equal(
      cbind(x, y),
      nv_array(cbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("stacks two 1-D vectors as columns (jit)", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3))
        y <- nv_array(c(4, 5, 6))
        cbind(x, y)
      },
      nv_array(cbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("stacks two matrices horizontally", {
    a <- matrix(1:6, nrow = 3)
    b <- matrix(7:12, nrow = 3)
    expect_jit_equal(
      {
        x <- nv_array(a)
        y <- nv_array(b)
        cbind(x, y)
      },
      nv_array(cbind(a, b))
    )
  })

  it("treats 1-D operand as a column when mixed with a matrix", {
    a <- matrix(1:6, nrow = 3)
    v <- c(7, 8, 9)
    expect_jit_equal(
      {
        x <- nv_array(a)
        y <- nv_array(v)
        cbind(x, y)
      },
      nv_array(cbind(a, v))
    )
  })

  it("accepts more than two arguments", {
    expect_jit_equal(
      {
        cbind(nv_array(c(1, 2)), nv_array(c(3, 4)), nv_array(c(5, 6)))
      },
      nv_array(cbind(c(1, 2), c(3, 4), c(5, 6)))
    )
  })

  it("errors when number of rows mismatch", {
    expect_error(cbind(nv_array(c(1, 2)), nv_array(c(3, 4, 5))))
  })

  it("nv_cbind matches cbind", {
    x <- nv_array(c(1, 2, 3))
    y <- nv_array(c(4, 5, 6))
    expect_equal(nv_cbind(x, y), cbind(x, y))
  })
})
