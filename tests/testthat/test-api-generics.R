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

  it("broadcasts a scalar to the column count of a matrix", {
    a <- matrix(1:6, nrow = 2)
    expect_equal(rbind(nv_array(a), nv_array(0)), nv_array(rbind(a, 0)))
    expect_equal(rbind(nv_array(0), nv_array(a)), nv_array(rbind(0, a)))
  })

  it("broadcasts a scalar to the column count of a 1-D vector", {
    v <- c(7, 8, 9)
    expect_equal(rbind(nv_array(v), nv_array(0)), nv_array(rbind(v, 0)))
  })

  it("treats all scalars as 1x1 rows", {
    expect_equal(rbind(nv_array(1), nv_array(2)), nv_array(rbind(1, 2)))
  })

  it("broadcasts a scalar against a 3-D array", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    out <- rbind(nv_array(a), nv_array(0))
    expect_equal(shape(out), c(3L, 3L, 4L))
    expect_equal(as_array(out)[1:2, , ], a)
    expect_equal(as_array(out)[3L, , ], array(0, dim = c(3L, 4L)))
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

  it("broadcasts a scalar to the row count of a matrix", {
    a <- matrix(1:6, nrow = 3)
    expect_equal(cbind(nv_array(a), nv_array(0)), nv_array(cbind(a, 0)))
    expect_equal(cbind(nv_array(0), nv_array(a)), nv_array(cbind(0, a)))
  })

  it("broadcasts a scalar to the row count of a 1-D vector", {
    v <- c(7, 8, 9)
    expect_equal(cbind(nv_array(v), nv_array(0)), nv_array(cbind(v, 0)))
  })

  it("treats all scalars as 1x1 columns", {
    expect_equal(cbind(nv_array(1), nv_array(2)), nv_array(cbind(1, 2)))
  })

  it("broadcasts a scalar against a 3-D array", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    out <- cbind(nv_array(a), nv_array(0))
    expect_equal(shape(out), c(2L, 4L, 4L))
    expect_equal(as_array(out)[, 1:3, ], a)
    expect_equal(as_array(out)[, 4L, ], array(0, dim = c(2L, 4L)))
  })
})

describe("as.double", {
  it("returns a bare double vector and discards shape", {
    x <- nv_array(c(1, 2, 3, 4, 5, 6), dtype = "f32", shape = c(2L, 3L))
    result <- as.double(x)
    expect_type(result, "double")
    expect_null(dim(result))
    expect_equal(result, c(1, 2, 3, 4, 5, 6))
  })

  it("is equivalent to as.numeric", {
    x <- nv_array(c(1.5, 2.5), dtype = "f64")
    expect_identical(as.numeric(x), as.double(x))
  })

  it("works on a scalar", {
    expect_identical(as.double(nv_scalar(3.5, dtype = "f64")), 3.5)
  })

  it("works on signed integer dtypes", {
    x <- nv_array(1:4, dtype = "i32", shape = c(2L, 2L))
    result <- as.double(x)
    expect_type(result, "double")
    expect_null(dim(result))
    expect_equal(result, c(1, 2, 3, 4))
  })

  it("works on unsigned integer dtypes", {
    expect_identical(as.double(nv_array(c(1L, 2L, 3L), dtype = "ui32")), c(1, 2, 3))
  })

  it("errors on bool dtype", {
    expect_error(as.double(nv_array(TRUE, dtype = "bool")), "requires a float or integer dtype")
  })
})

describe("as.integer", {
  it("returns a bare integer vector and discards shape", {
    x <- nv_array(1:4, dtype = "i32", shape = c(2L, 2L))
    result <- as.integer(x)
    expect_type(result, "integer")
    expect_null(dim(result))
    expect_equal(result, 1:4)
  })

  it("works on unsigned integer dtypes", {
    x <- nv_array(c(1L, 2L, 3L), dtype = "ui32")
    expect_identical(as.integer(x), c(1L, 2L, 3L))
  })

  it("works on a scalar", {
    expect_identical(as.integer(nv_scalar(7L, dtype = "i32")), 7L)
  })

  it("errors on non-integer dtype", {
    expect_error(as.integer(nv_array(1.5, dtype = "f32")), "requires a .* integer dtype")
    expect_error(as.integer(nv_array(TRUE, dtype = "bool")), "requires a .* integer dtype")
  })
})

describe("as.logical", {
  it("returns a bare logical vector and discards shape", {
    x <- nv_array(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool", shape = c(2L, 2L))
    result <- as.logical(x)
    expect_type(result, "logical")
    expect_null(dim(result))
    expect_equal(result, c(TRUE, FALSE, TRUE, FALSE))
  })

  it("errors on non-bool dtype", {
    expect_error(as.logical(nv_array(1L, dtype = "i32")), "requires a .*bool.* dtype")
    expect_error(as.logical(nv_array(1.5, dtype = "f32")), "requires a .*bool.* dtype")
  })
})
