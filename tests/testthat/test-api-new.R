describe("nv_log2", {
  it("computes base-2 logarithm", {
    expect_jit_equal(
      nv_log2(nv_tensor(c(1, 2, 4, 8))),
      nv_tensor(log2(c(1, 2, 4, 8))),
      tolerance = 1e-6
    )
  })
  it("works on scalars", {
    expect_jit_equal(
      nv_log2(nv_scalar(16)),
      nv_scalar(4),
      tolerance = 1e-6
    )
  })
})

describe("nv_log10", {
  it("computes base-10 logarithm", {
    expect_jit_equal(
      nv_log10(nv_tensor(c(1, 10, 100, 1000))),
      nv_tensor(log10(c(1, 10, 100, 1000))),
      tolerance = 1e-6
    )
  })
  it("works on scalars", {
    expect_jit_equal(
      nv_log10(nv_scalar(1000)),
      nv_scalar(3),
      tolerance = 1e-6
    )
  })
})

describe("nv_is_nan", {
  it("detects NaN values", {
    expect_jit_equal(
      nv_is_nan(nv_tensor(c(1, NaN, Inf, -Inf, 0))),
      nv_tensor(c(FALSE, TRUE, FALSE, FALSE, FALSE))
    )
  })
  it("works via is.nan() generic", {
    expect_jit_equal(
      is.nan(nv_tensor(c(1, NaN, Inf, -Inf, 0))),
      nv_tensor(c(FALSE, TRUE, FALSE, FALSE, FALSE))
    )
  })
})

describe("nv_is_infinite", {
  it("detects infinite values", {
    expect_jit_equal(
      nv_is_infinite(nv_tensor(c(1, NaN, Inf, -Inf, 0))),
      nv_tensor(c(FALSE, FALSE, TRUE, TRUE, FALSE))
    )
  })
  it("works via is.infinite() generic", {
    expect_jit_equal(
      is.infinite(nv_tensor(c(1, NaN, Inf, -Inf, 0))),
      nv_tensor(c(FALSE, FALSE, TRUE, TRUE, FALSE))
    )
  })
})

describe("nv_var", {
  it("computes variance with Bessel's correction", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_jit_equal(
      nv_var(nv_tensor(vals), dims = 1L),
      nv_scalar(var(vals)),
      tolerance = 1e-5
    )
  })
  it("computes population variance with correction = 0", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expected <- mean((vals - mean(vals))^2)
    expect_jit_equal(
      nv_var(nv_tensor(vals), dims = 1L, correction = 0L),
      nv_scalar(expected),
      tolerance = 1e-5
    )
  })
  it("works along specific dimensions of a matrix", {
    vals <- c(1, 2, 3, 4, 5, 6)
    m <- matrix(vals, nrow = 2)
    expect_jit_equal(
      nv_var(nv_tensor(vals, shape = c(2, 3), dtype = "f32"), dims = 2L),
      nv_tensor(apply(m, 1, var), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("nv_sd", {
  it("computes standard deviation", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_jit_equal(
      nv_sd(nv_tensor(vals), dims = 1L),
      nv_scalar(sd(vals)),
      tolerance = 1e-5
    )
  })
})

describe("nv_squeeze", {
  it("removes all size-1 dimensions by default", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:6, shape = c(1, 6, 1))
        nv_squeeze(x)
      },
      nv_tensor(1:6, shape = 6L)
    )
  })
  it("removes specific dimensions", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:6, shape = c(1, 6, 1))
        nv_squeeze(x, dims = 1L)
      },
      nv_tensor(1:6, shape = c(6, 1))
    )
  })
  it("errors when squeezing non-1 dimension", {
    expect_error(
      jit_eval(nv_squeeze(nv_tensor(1:6, shape = c(2, 3)), dims = 1L)),
      "Cannot squeeze"
    )
  })
})

describe("nv_unsqueeze", {
  it("adds dimension at the beginning", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3))
        nv_unsqueeze(x, dim = 1L)
      },
      nv_tensor(c(1, 2, 3), shape = c(1, 3))
    )
  })
  it("adds dimension at the end", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3))
        nv_unsqueeze(x, dim = 2L)
      },
      nv_tensor(c(1, 2, 3), shape = c(3, 1))
    )
  })
  it("adds dimension in the middle", {
    x <- nv_tensor(1:6, shape = c(2, 3))
    result <- jit(\() nv_unsqueeze(x, dim = 2L))()
    expect_equal(shape(result), c(2L, 1L, 3L))
    roundtrip <- jit(\() nv_squeeze(nv_unsqueeze(x, dim = 2L), dims = 2L))()
    expect_equal(roundtrip, x)
  })
})

describe("nv_expand", {
  it("broadcasts to target shape", {
    result <- jit_eval(nv_expand(nv_tensor(c(1, 2, 3)), shape = c(2, 3)))
    expect_equal(shape(result), c(2L, 3L))
  })
})

describe("nv_flip", {
  it("reverses along a dimension", {
    expect_jit_equal(
      nv_flip(nv_tensor(c(1, 2, 3, 4, 5)), dims = 1L),
      nv_tensor(c(5, 4, 3, 2, 1))
    )
  })
})

describe("nv_linspace", {
  it("creates evenly spaced values", {
    expect_jit_equal(
      nv_linspace(0, 1, steps = 5L),
      nv_tensor(c(0, 0.25, 0.5, 0.75, 1)),
      tolerance = 1e-6
    )
  })
  it("handles single step", {
    expect_jit_equal(
      nv_linspace(3, 7, steps = 1L),
      nv_tensor(3, shape = 1L),
      tolerance = 1e-6
    )
  })
  it("works with integer-like endpoints", {
    expect_jit_equal(
      nv_linspace(0, 10, steps = 6L),
      nv_tensor(c(0, 2, 4, 6, 8, 10)),
      tolerance = 1e-6
    )
  })
})

describe("nv_outer", {
  it("computes outer product", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3))
        y <- nv_tensor(c(4, 5))
        nv_outer(x, y)
      },
      nv_tensor(c(4, 8, 12, 5, 10, 15), shape = c(3, 2)),
      tolerance = 1e-6
    )
  })
  it("promotes types", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1L, 2L))
        y <- nv_tensor(c(1.5, 2.5))
        nv_outer(x, y)
      },
      nv_tensor(c(1.5, 3, 2.5, 5), shape = c(2, 2)),
      tolerance = 1e-6
    )
  })
})

describe("nv_extract_diag", {
  it("extracts diagonal from square matrix", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3, 4, 5, 6, 7, 8, 9), shape = c(3, 3), dtype = "f32")
        nv_extract_diag(x)
      },
      nv_tensor(c(1, 5, 9), dtype = "f32")
    )
  })
  it("extracts diagonal from rectangular matrix", {
    expect_jit_equal(
      {
        x <- nv_tensor(1:6, shape = c(2, 3), dtype = "f32")
        nv_extract_diag(x)
      },
      nv_tensor(c(1, 4), dtype = "f32")
    )
  })
})

describe("nv_trace", {
  it("computes trace of a matrix", {
    expect_jit_equal(
      nv_trace(nv_tensor(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))),
      nv_scalar(6),
      tolerance = 1e-6
    )
  })
  it("computes trace of identity", {
    expect_jit_equal(
      nv_trace(nv_eye(4L)),
      nv_scalar(4),
      tolerance = 1e-6
    )
  })
})

describe("nv_tril", {
  it("returns lower triangular part", {
    result <- jit_eval(nv_tril(nv_fill(1, c(3, 3))))
    expected <- matrix(c(1, 1, 1, 0, 1, 1, 0, 0, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports positive diagonal offset", {
    result <- jit_eval(nv_tril(nv_fill(1, c(3, 3)), diagonal = 1L))
    expected <- matrix(c(1, 1, 1, 1, 1, 1, 0, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports negative diagonal offset", {
    result <- jit_eval(nv_tril(nv_fill(1, c(3, 3)), diagonal = -1L))
    expected <- matrix(c(0, 1, 1, 0, 0, 1, 0, 0, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
})

describe("nv_triu", {
  it("returns upper triangular part", {
    result <- jit_eval(nv_triu(nv_fill(1, c(3, 3))))
    expected <- matrix(c(1, 0, 0, 1, 1, 0, 1, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports positive diagonal offset", {
    result <- jit_eval(nv_triu(nv_fill(1, c(3, 3)), diagonal = 1L))
    expected <- matrix(c(0, 0, 0, 1, 0, 0, 1, 1, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports negative diagonal offset", {
    result <- jit_eval(nv_triu(nv_fill(1, c(3, 3)), diagonal = -1L))
    expected <- matrix(c(1, 1, 0, 1, 1, 1, 1, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
})

describe("nv_crossprod", {
  it("computes t(x) %*% y", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        y <- nv_tensor(c(7, 8, 9, 10, 11, 12), shape = c(3, 2), dtype = "f32")
        nv_crossprod(x, y)
      },
      nv_tensor(as.numeric(crossprod(matrix(1:6, 3, 2), matrix(7:12, 3, 2))),
                shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("computes t(x) %*% x when y is NULL", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        nv_crossprod(x)
      },
      nv_tensor(as.numeric(crossprod(matrix(1:6, 3, 2))),
                shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("works via S3 generic", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        crossprod(x)
      },
      nv_tensor(as.numeric(crossprod(matrix(1:6, 3, 2))),
                shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("nv_tcrossprod", {
  it("computes x %*% t(y)", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        y <- nv_tensor(c(7, 8, 9, 10, 11, 12), shape = c(2, 3), dtype = "f32")
        nv_tcrossprod(x, y)
      },
      nv_tensor(as.numeric(tcrossprod(matrix(1:6, 2, 3), matrix(7:12, 2, 3))),
                shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("computes x %*% t(x) when y is NULL", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        nv_tcrossprod(x)
      },
      nv_tensor(as.numeric(tcrossprod(matrix(1:6, 2, 3))),
                shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("works via S3 generic", {
    expect_jit_equal(
      {
        x <- nv_tensor(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        tcrossprod(x)
      },
      nv_tensor(as.numeric(tcrossprod(matrix(1:6, 2, 3))),
                shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})
