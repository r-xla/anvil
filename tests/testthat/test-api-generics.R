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

describe("solve", {
  it("solves a linear system", {
    a_mat <- matrix(c(4, 2, 2, 3), nrow = 2)
    b_mat <- matrix(c(1, 2), nrow = 2)
    expect_jit_equal(
      solve(nv_array(a_mat, dtype = "f32"), nv_array(b_mat, dtype = "f32")),
      nv_array(base::solve(a_mat, b_mat), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("rev", {
  it("reverses a 1-D arrayish", {
    vals <- c(1, 2, 3, 4, 5)
    expect_jit_equal(
      rev(nv_array(vals, dtype = "f32")),
      nv_array(rev(vals), dtype = "f32")
    )
  })
  it("errors for arrayish with rank > 1", {
    x <- nv_array(matrix(1:6, nrow = 2), dtype = "f32")
    expect_error(rev(x), "only defined for 1-D")
  })
})

describe("%/%", {
  it("computes integer (floor) division", {
    a <- c(7, -7, 6, -6)
    b <- c(2, 2, 3, 3)
    expect_jit_equal(
      nv_array(a, dtype = "f32") %/% nv_array(b, dtype = "f32"),
      nv_array(a %/% b, dtype = "f32")
    )
  })
})

describe("range", {
  it("returns c(min, max) over all dimensions", {
    vals <- c(3, 1, 4, 1, 5, 9, 2, 6)
    expect_jit_equal(
      range(nv_array(vals, dtype = "f32")),
      nv_array(range(vals), dtype = "f32")
    )
  })
})

describe("head", {
  it("takes the first n elements of a 1-D arrayish", {
    vals <- 1:10
    expect_jit_equal(
      head(nv_array(vals, dtype = "f32"), n = 3L),
      nv_array(head(vals, 3L), dtype = "f32")
    )
  })
  it("takes the first n rows of a 2-D arrayish", {
    m <- matrix(1:12, nrow = 4)
    expect_jit_equal(
      head(nv_array(m, dtype = "f32"), n = 2L),
      nv_array(head(m, 2L), dtype = "f32")
    )
  })
})

describe("tail", {
  it("takes the last n elements of a 1-D arrayish", {
    vals <- 1:10
    expect_jit_equal(
      tail(nv_array(vals, dtype = "f32"), n = 3L),
      nv_array(tail(vals, 3L), dtype = "f32")
    )
  })
  it("takes the last n rows of a 2-D arrayish", {
    m <- matrix(1:12, nrow = 4)
    expect_jit_equal(
      tail(nv_array(m, dtype = "f32"), n = 2L),
      nv_array(tail(m, 2L), dtype = "f32")
    )
  })
})

describe("aperm", {
  it("permutes dims with explicit perm", {
    arr <- array(seq_len(24), dim = c(2, 3, 4))
    expect_jit_equal(
      aperm(nv_array(arr, dtype = "f32"), perm = c(3L, 1L, 2L)),
      nv_array(aperm(arr, perm = c(3L, 1L, 2L)), dtype = "f32")
    )
  })
  it("reverses dims when perm is NULL", {
    arr <- array(seq_len(24), dim = c(2, 3, 4))
    expect_jit_equal(
      aperm(nv_array(arr, dtype = "f32")),
      nv_array(aperm(arr), dtype = "f32")
    )
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
