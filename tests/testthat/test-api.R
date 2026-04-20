test_that("auto-broadcasting higher-dimensional arrays is not supported (it's bug prone)", {
  x <- nv_array(1:2, shape = c(2, 1))
  y <- nv_array(1:2, shape = c(1, 2))
  expect_error(
    jit(nv_add)(x, y),
    "must have the same shape"
  )
})

test_that("broadcasting scalars", {
  fjit <- jit(nv_add)
  expect_equal(
    fjit(
      nv_scalar(1),
      nv_array(0, shape = c(2, 2))
    ),
    nv_array(1, shape = c(2, 2))
  )
})

test_that("infix add", {
  f <- jit(function(x, y) {
    x + y
  })
  expect_equal(
    f(
      nv_array(1, shape = c(2, 2)),
      nv_array(0, shape = c(2, 2))
    ),
    nv_array(1, shape = c(2, 2))
  )
})

test_that("jit constant single return is bare array", {
  f <- jit(function() nv_scalar(0.5))
  out <- f()
  expect_equal(as_array(out), 0.5, tolerance = 1e-6)
})

test_that("Summary group generics", {
  fsum <- jit(function(x) sum(x))
  expect_equal(as_array(fsum(nv_array(1:10))), 55)
})

test_that("mean", {
  fmean <- jit(function(x) mean(x))
  expect_equal(as_array(fmean(nv_array(1:10, "f32"))), 5.5)
})

test_that("constants can be lifted to the appropriate level", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_equal(
    jit(gradient(f, wrt = "x"))(nv_scalar(2))[[1L]],
    nv_scalar(1)
  )
})

test_that("wrt non-existent argument", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_error(
    jit(gradient(f, wrt = "y"))(nv_array(2)),
    "must be a subset"
  )
})

test_that("promote to common", {
  f <- function(x, y) {
    nv_add(x, y)
  }
  expect_equal(
    jit(f)(nv_array(1, dtype = "i32"), nv_array(1.0, dtype = "f32")),
    nv_array(2.0, dtype = "f32")
  )
})

test_that("nv_clamp converts min and max to operand dtype", {
  expect_equal(
    jit_eval(nv_clamp(nv_scalar(0L), nv_array(c(-1, 0.5, 2), dtype = "f32"), nv_scalar(1L))),
    nv_array(c(0, 0.5, 1), dtype = "f32")
  )
})

describe("nv_concatenate", {
  it("auto-promotes to common", {
    expect_equal(
      jit_eval(nv_concatenate(nv_array(c(1, 2)), nv_array(3:4))),
      nv_array(c(1, 2, 3, 4))
    )
  })
  it("can concatenate literals", {
    # Pure literals produce ambiguous output
    expect_equal(
      jit_eval(nv_concatenate(1L, 2L)),
      nv_array(1:2, ambiguous = TRUE)
    )
    expect_equal(
      jit_eval(nv_concatenate(1L, 2L, dimension = 1L)),
      nv_array(1:2, ambiguous = TRUE)
    )
    # Mixed array + literal: non-ambiguous array determines output ambiguity
    expect_equal(
      jit_eval(nv_concatenate(nv_array(1:2), 3L)),
      nv_array(1:3)
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_array(1L), 2L)),
      nv_array(1:2)
    )
  })
  it("fails when dimension is out of bounds", {
    expect_error(
      jit_eval(nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), dimension = 3L))
    )
  })
  it("can concatenate 2d arrays", {
    expect_equal(
      jit_eval(nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), dimension = 1L)),
      nv_array(1:4, shape = c(4, 1))
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), dimension = 2L)),
      nv_array(1:4, shape = c(2, 2), dtype = "i32")
    )
  })
  it("fails with incompatible shapes", {
    expect_error(
      jit_eval(nv_concatenate(nv_array(1, shape = c(1, 1, 1)), nv_array(2, shape = c(1, 1)), dimension = 1L))
    )
  })
})

describe("nv_log2", {
  it("computes base-2 logarithm", {
    expect_jit_equal(
      nv_log2(nv_array(c(1, 2, 4, 8))),
      nv_array(log2(c(1, 2, 4, 8))),
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
      nv_log10(nv_array(c(1, 10, 100, 1000))),
      nv_array(log10(c(1, 10, 100, 1000))),
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

describe("nv_is_finite", {
  it("detects finite values", {
    expect_jit_equal(
      nv_is_finite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(TRUE, FALSE, FALSE, FALSE, TRUE))
    )
  })
  it("works via is.finite() generic", {
    expect_jit_equal(
      is.finite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(TRUE, FALSE, FALSE, FALSE, TRUE))
    )
  })
})

describe("nv_is_nan", {
  it("detects NaN values", {
    expect_jit_equal(
      nv_is_nan(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, TRUE, FALSE, FALSE, FALSE))
    )
  })
  it("works via is.nan() generic", {
    expect_jit_equal(
      is.nan(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, TRUE, FALSE, FALSE, FALSE))
    )
  })
})

describe("nv_is_infinite", {
  it("detects infinite values", {
    expect_jit_equal(
      nv_is_infinite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, FALSE, TRUE, TRUE, FALSE))
    )
  })
  it("works via is.infinite() generic", {
    expect_jit_equal(
      is.infinite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, FALSE, TRUE, TRUE, FALSE))
    )
  })
})

describe("nv_var", {
  it("computes variance with Bessel's correction", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_jit_equal(
      nv_var(nv_array(vals), dims = 1L),
      nv_scalar(var(vals)),
      tolerance = 1e-5
    )
  })
  it("computes population variance with correction = 0", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expected <- mean((vals - mean(vals))^2)
    expect_jit_equal(
      nv_var(nv_array(vals), dims = 1L, correction = 0L),
      nv_scalar(expected),
      tolerance = 1e-5
    )
  })
  it("works along specific dimensions of a matrix", {
    vals <- c(1, 2, 3, 4, 5, 6)
    m <- matrix(vals, nrow = 2)
    expect_jit_equal(
      nv_var(nv_array(vals, shape = c(2, 3), dtype = "f32"), dims = 2L),
      nv_array(apply(m, 1, var), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("nv_sd", {
  it("computes standard deviation", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_jit_equal(
      nv_sd(nv_array(vals), dims = 1L),
      nv_scalar(sd(vals)),
      tolerance = 1e-5
    )
  })
})

describe("nv_squeeze", {
  it("removes all size-1 dimensions by default", {
    expect_jit_equal(
      {
        x <- nv_array(1:6, shape = c(1, 6, 1))
        nv_squeeze(x)
      },
      nv_array(1:6, shape = 6L)
    )
  })
  it("removes specific dimensions", {
    expect_jit_equal(
      {
        x <- nv_array(1:6, shape = c(1, 6, 1))
        nv_squeeze(x, dims = 1L)
      },
      nv_array(1:6, shape = c(6, 1))
    )
  })
  it("errors when squeezing non-1 dimension", {
    expect_error(
      jit_eval(nv_squeeze(nv_array(1:6, shape = c(2, 3)), dims = 1L)),
      "Cannot squeeze"
    )
  })
})

describe("nv_unsqueeze", {
  it("adds dimension at the beginning", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3))
        nv_unsqueeze(x, dim = 1L)
      },
      nv_array(c(1, 2, 3), shape = c(1, 3))
    )
  })
  it("adds dimension at the end", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3))
        nv_unsqueeze(x, dim = 2L)
      },
      nv_array(c(1, 2, 3), shape = c(3, 1))
    )
  })
  it("adds dimension in the middle", {
    x <- nv_array(1:6, shape = c(2, 3))
    result <- jit(\() nv_unsqueeze(x, dim = 2L))()
    expect_equal(shape(result), c(2L, 1L, 3L))
    roundtrip <- jit(\() nv_squeeze(nv_unsqueeze(x, dim = 2L), dims = 2L))()
    expect_equal(roundtrip, x)
  })
})

describe("nv_seq with steps", {
  it("creates evenly spaced values", {
    expect_jit_equal(
      nv_seq(0, 1, steps = 5L),
      nv_array(c(0, 0.25, 0.5, 0.75, 1)),
      tolerance = 1e-6
    )
  })
  it("handles single step", {
    expect_jit_equal(
      nv_seq(3, 7, steps = 1L),
      nv_array(3, shape = 1L),
      tolerance = 1e-6
    )
  })
  it("works with integer-like endpoints", {
    expect_jit_equal(
      nv_seq(0, 10, steps = 6L),
      nv_array(c(0, 2, 4, 6, 8, 10)),
      tolerance = 1e-6
    )
  })
})

describe("nv_outer", {
  it("computes outer product", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3))
        y <- nv_array(c(4, 5))
        nv_outer(x, y)
      },
      nv_array(c(4, 8, 12, 5, 10, 15), shape = c(3, 2)),
      tolerance = 1e-6
    )
  })
  it("promotes types", {
    expect_jit_equal(
      {
        x <- nv_array(c(1L, 2L))
        y <- nv_array(c(1.5, 2.5))
        nv_outer(x, y)
      },
      nv_array(c(1.5, 3, 2.5, 5), shape = c(2, 2)),
      tolerance = 1e-6
    )
  })
})

describe("nv_extract_diag", {
  it("extracts diagonal from square matrix", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6, 7, 8, 9), shape = c(3, 3), dtype = "f32")
        nv_extract_diag(x)
      },
      nv_array(c(1, 5, 9), dtype = "f32")
    )
  })
  it("extracts diagonal from rectangular matrix", {
    expect_jit_equal(
      {
        x <- nv_array(1:6, shape = c(2, 3), dtype = "f32")
        nv_extract_diag(x)
      },
      nv_array(c(1, 4), dtype = "f32")
    )
  })
})

describe("nv_trace", {
  it("computes trace of a matrix", {
    expect_jit_equal(
      nv_trace(nv_array(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))),
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

describe("nv_diag", {
  it("builds a diagonal matrix from a 1-D array", {
    result <- nv_diag(nv_array(c(1, 2, 3)))
    expected <- diag(c(1, 2, 3))
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("works under jit (device() on a GraphBox operand)", {
    f <- jit(function(x) nv_diag(x))
    expect_equal(
      as_array(f(nv_array(c(1, 2, 3)))),
      diag(c(1, 2, 3)),
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
  it("works under jit (device() on a GraphBox operand)", {
    f <- jit(function(x) nv_tril(x, diagonal = -1L))
    expected <- matrix(c(0, 1, 1, 0, 0, 1, 0, 0, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(f(nv_fill(1, c(3, 3)))), expected, tolerance = 1e-6)
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
  it("works under jit (device() on a GraphBox operand)", {
    f <- jit(function(x) nv_triu(x, diagonal = 1L))
    expected <- matrix(c(0, 0, 0, 1, 0, 0, 1, 1, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(f(nv_fill(1, c(3, 3)))), expected, tolerance = 1e-6)
  })
})

describe("nv_tril with quickr backend", {
  it("works when operand is quickr", {
    skip_if_not_installed("quickr")
    x <- nv_array(matrix(1, 3, 3), backend = "quickr")
    result <- nv_tril(x)
    expected <- matrix(c(1, 1, 1, 0, 1, 1, 0, 0, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
})

describe("nv_triu with quickr backend", {
  it("works when operand is quickr", {
    skip_if_not_installed("quickr")
    x <- nv_array(matrix(1, 3, 3), backend = "quickr")
    result <- nv_triu(x)
    expected <- matrix(c(1, 0, 0, 1, 1, 0, 1, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
})

describe("nv_crossprod", {
  it("computes t(x) %*% y", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        y <- nv_array(c(7, 8, 9, 10, 11, 12), shape = c(3, 2), dtype = "f32")
        nv_crossprod(x, y)
      },
      nv_array(as.numeric(crossprod(matrix(1:6, 3, 2), matrix(7:12, 3, 2))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("computes t(x) %*% x when y is NULL", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        nv_crossprod(x)
      },
      nv_array(as.numeric(crossprod(matrix(1:6, 3, 2))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("works via S3 generic", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        crossprod(x)
      },
      nv_array(as.numeric(crossprod(matrix(1:6, 3, 2))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("nv_tcrossprod", {
  it("computes x %*% t(y)", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        y <- nv_array(c(7, 8, 9, 10, 11, 12), shape = c(2, 3), dtype = "f32")
        nv_tcrossprod(x, y)
      },
      nv_array(as.numeric(tcrossprod(matrix(1:6, 2, 3), matrix(7:12, 2, 3))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("computes x %*% t(x) when y is NULL", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        nv_tcrossprod(x)
      },
      nv_array(as.numeric(tcrossprod(matrix(1:6, 2, 3))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("works via S3 generic", {
    expect_jit_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        tcrossprod(x)
      },
      nv_array(as.numeric(tcrossprod(matrix(1:6, 2, 3))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("nv_fill_like", {
  it("inherits shape, dtype, ambiguous, device from like", {
    like <- nv_array(matrix(1:6, nrow = 2), dtype = "i16")
    out <- nv_fill_like(like, 0L)
    expect_equal(shape(out), shape(like))
    expect_equal(dtype(out), dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
    expect_equal(as.integer(as_array(out)), rep(0L, 6L))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_array(matrix(1:6, nrow = 2), dtype = "i16")
    out <- nv_fill_like(like, 1, shape = 5L, dtype = "f32")
    expect_equal(shape(out), 5L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_iota_like", {
  it("inherits shape, dtype, ambiguous, device from like", {
    like <- nv_array(matrix(0L, nrow = 2, ncol = 3), dtype = "i16")
    out <- nv_iota_like(like, dim = 1L)
    expect_equal(shape(out), shape(like))
    expect_equal(dtype(out), dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_array(matrix(0L, nrow = 2, ncol = 3), dtype = "i16")
    out <- nv_iota_like(like, dim = 1L, shape = 4L, dtype = "i32")
    expect_equal(shape(out), 4L)
    expect_equal(dtype(out), as_dtype("i32"))
  })
})

describe("nv_seq_like", {
  it("inherits dtype, ambiguous, device from like (length determined by start/end)", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    out <- nv_seq_like(like, 1L, 5L)
    expect_equal(dtype(out), dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
    expect_equal(shape(out), 5L)
    expect_equal(as.integer(as_array(out)), c(1L, 2L, 3L, 4L, 5L))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    out <- nv_seq_like(like, 1, 5, dtype = "f32")
    expect_equal(dtype(out), as_dtype("f32"))
  })
})
