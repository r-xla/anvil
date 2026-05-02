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

describe("nv_solve", {
  # nv_solve switched from Cholesky (SPD-only) to LU (general non-singular).
  # The LU path goes via nvl_lu + an in-graph permutation loop +
  # two triangular solves, so these tests exercise that whole pipeline.

  it("solves an SPD system (regression: old Cholesky path)", {
    A <- matrix(c(4, 2, 2, 3), nrow = 2)
    b <- c(1, 2)
    expected <- nv_array(as.numeric(solve(A, b)), dtype = "f64")
    expect_jit_equal(
      nv_solve(nv_array(A, dtype = "f64"), nv_array(b, dtype = "f64")),
      expected,
      tolerance = 1e-12
    )
  })

  it("solves a general (non-symmetric) system", {
    set.seed(1)
    A <- matrix(rnorm(9), nrow = 3)
    b <- c(1, 2, 3)
    expected <- nv_array(as.numeric(solve(A, b)), dtype = "f64")
    expect_jit_equal(
      nv_solve(nv_array(A, dtype = "f64"), nv_array(b, dtype = "f64")),
      expected,
      tolerance = 1e-12
    )
  })

  it("solves a system that requires row pivoting", {
    # Zero in [1,1] forces getrf to swap rows on the first elimination step.
    A <- matrix(c(0, 1, 1, 1), nrow = 2)
    b <- c(1, 2)
    expected <- nv_array(as.numeric(solve(A, b)), dtype = "f64")
    expect_jit_equal(
      nv_solve(nv_array(A, dtype = "f64"), nv_array(b, dtype = "f64")),
      expected,
      tolerance = 1e-12
    )
  })

  it("solves with multiple right-hand sides", {
    set.seed(2)
    A <- matrix(rnorm(9), nrow = 3)
    B <- matrix(rnorm(6), nrow = 3)
    expected <- nv_array(solve(A, B), dtype = "f64")
    expect_jit_equal(
      nv_solve(nv_array(A, dtype = "f64"), nv_array(B, dtype = "f64")),
      expected,
      tolerance = 1e-12
    )
  })

  it("solves a larger system", {
    set.seed(3)
    n <- 10L
    A <- matrix(rnorm(n * n), nrow = n) + diag(n) * 0.1
    b <- rnorm(n)
    expected <- nv_array(as.numeric(solve(A, b)), dtype = "f64")
    expect_jit_equal(
      nv_solve(nv_array(A, dtype = "f64"), nv_array(b, dtype = "f64")),
      expected,
      tolerance = 1e-10
    )
  })

  it("works in f32", {
    set.seed(4)
    A <- matrix(rnorm(9), nrow = 3)
    b <- rnorm(3)
    expected <- nv_array(as.numeric(solve(A, b)), dtype = "f32")
    expect_jit_equal(
      nv_solve(nv_array(A, dtype = "f32"), nv_array(b, dtype = "f32")),
      expected,
      tolerance = 1e-4
    )
  })

  it("rejects non-square a", {
    A <- nv_array(matrix(rnorm(6), nrow = 2), dtype = "f64")
    b <- nv_array(c(1, 2), dtype = "f64")
    expect_error(nv_solve(A, b), "square")
  })

  it("rejects b with mismatched first dimension", {
    A <- nv_array(matrix(rnorm(9), nrow = 3), dtype = "f64")
    b <- nv_array(c(1, 2), dtype = "f64")
    expect_error(nv_solve(A, b), "length 3|3 rows")
  })
})

describe("nv_det / nv_logdet / nv_inv", {
  # All three are pure-R compositions on top of the new nvl_lu primitive.
  # We compare against base R's det() / determinant() / solve() respectively.
  # The det/logdet outputs are 0-D scalars, so unwrap with as.numeric()
  # before comparing.

  scalar_jit <- function(.expr) {
    expr <- substitute(.expr)
    eval_env <- new.env(parent = parent.frame())
    as.numeric(as_array(jit(\() eval(expr, envir = eval_env))()))
  }

  it("nv_det matches base::det on random matrices", {
    set.seed(1)
    for (n in c(2L, 3L, 5L)) {
      A <- matrix(rnorm(n * n), nrow = n) + diag(n) * 0.5
      expect_equal(
        scalar_jit(nv_det(nv_array(A, dtype = "f64"))),
        det(A),
        tolerance = 1e-12,
        info = paste0("n = ", n)
      )
    }
  })

  it("nv_det handles a forced row pivot (negative determinant)", {
    # getrf swaps rows here; the implementation must multiply by sign(P).
    A <- matrix(c(0, 1, 1, 1), nrow = 2)
    expect_equal(
      scalar_jit(nv_det(nv_array(A, dtype = "f64"))),
      det(A),
      tolerance = 1e-12
    )
  })

  it("nv_det matches base::det in f32", {
    set.seed(2)
    A <- matrix(rnorm(16), 4) + diag(4) * 0.5
    expect_equal(
      scalar_jit(nv_det(nv_array(A, dtype = "f32"))),
      det(A),
      tolerance = 1e-4
    )
  })

  it("nv_logdet matches base::determinant", {
    set.seed(3)
    for (n in c(2L, 3L, 5L)) {
      # Add diagonal to avoid near-zero determinants where log is unstable.
      A <- matrix(rnorm(n * n), nrow = n) + diag(n) * 1.5
      expected <- as.numeric(determinant(A, logarithm = TRUE)$modulus)
      expect_equal(
        scalar_jit(nv_logdet(nv_array(A, dtype = "f64"))),
        expected,
        tolerance = 1e-12,
        info = paste0("n = ", n)
      )
    }
  })

  it("nv_inv matches base::solve", {
    set.seed(4)
    for (n in c(2L, 3L, 5L)) {
      A <- matrix(rnorm(n * n), nrow = n) + diag(n) * 0.5
      expected <- nv_array(solve(A), dtype = "f64")
      expect_jit_equal(
        nv_inv(nv_array(A, dtype = "f64")),
        expected,
        tolerance = 1e-10,
        info = paste0("n = ", n)
      )
    }
  })

  it("rejects non-square inputs", {
    A <- nv_array(matrix(rnorm(6), nrow = 2), dtype = "f64")
    expect_error(nv_det(A), "square")
    expect_error(nv_logdet(A), "square")
    expect_error(nv_inv(A), "square")
  })
})

describe("S3 linalg generics on AnvilArray", {
  # These cover the dispatch wiring from base R's S3 generics
  # (solve, qr, chol, determinant) to our nv_* implementations.

  set.seed(1)
  A_r <- matrix(rnorm(9), 3) + diag(3) * 0.5
  b_r <- c(1, 2, 3)
  spd_r <- crossprod(A_r)

  it("solve(A, b) calls nv_solve", {
    expect_jit_equal(
      solve(nv_array(A_r, dtype = "f64"), nv_array(b_r, dtype = "f64")),
      nv_array(as.numeric(solve(A_r, b_r)), dtype = "f64"),
      tolerance = 1e-12
    )
  })

  it("solve(A) (missing b) returns the inverse", {
    expect_jit_equal(
      solve(nv_array(A_r, dtype = "f64")),
      nv_array(solve(A_r), dtype = "f64"),
      tolerance = 1e-10
    )
  })

  it("qr(A) returns named (Q, R) and reconstructs A", {
    res <- jit_eval(qr(nv_array(A_r, dtype = "f64")))
    expect_named(res, c("Q", "R"))
    expect_equal(as_array(res$Q) %*% as_array(res$R), A_r, tolerance = 1e-12)
  })

  it("chol(SPD) matches base::chol (upper triangular)", {
    expect_jit_equal(
      chol(nv_array(spd_r, dtype = "f64")),
      nv_array(chol(spd_r), dtype = "f64"),
      tolerance = 1e-10
    )
  })

  it("determinant(A) matches base::determinant", {
    res <- jit_eval(determinant(nv_array(A_r, dtype = "f64"), logarithm = TRUE))
    base_d <- determinant(A_r, logarithm = TRUE)
    expect_equal(as.numeric(as_array(res$modulus)),
                 as.numeric(base_d$modulus), tolerance = 1e-12)
    expect_equal(as.numeric(as_array(res$sign)),
                 as.numeric(base_d$sign))
  })

  it("determinant(logarithm = FALSE) returns the absolute determinant", {
    res <- jit_eval(determinant(nv_array(A_r, dtype = "f64"), logarithm = FALSE))
    expect_equal(as.numeric(as_array(res$modulus)),
                 abs(det(A_r)), tolerance = 1e-12)
  })

  it("solve generic composes with %*% and t() in a single trace", {
    # Smoke check: a sequence of base R linalg generics all go through our
    # methods when given AnvilArray inputs. Use matrix RHS to keep matmul
    # in 2D.
    B_r <- matrix(b_r, ncol = 1L)
    expect_jit_equal(
      {
        a <- nv_array(spd_r, dtype = "f64")
        solve(a, t(a) %*% nv_array(B_r, dtype = "f64"))
      },
      nv_array(solve(spd_r, t(spd_r) %*% B_r), dtype = "f64"),
      tolerance = 1e-10
    )
  })

  it("nv_eigen() on a symmetric matrix matches base::eigen", {
    set.seed(20)
    M <- matrix(rnorm(16), 4)
    A <- (M + t(M)) / 2 + diag(seq_len(4))
    res <- jit_eval(nv_eigen(nv_array(A, dtype = "f64")))
    base_e <- base::eigen(A, symmetric = TRUE)
    expect_equal(as.numeric(as_array(res$values)),
                 base_e$values, tolerance = 1e-10)
    # Eigenvectors only match up to a sign per column; compare absolute
    # values column-wise.
    expect_equal(abs(as_array(res$vectors)),
                 abs(base_e$vectors), tolerance = 1e-10)
  })

  it("nv_eigen() returns descending-ordered eigenvalues (matches base R)", {
    A <- matrix(c(2, 1, 1, 2), 2)  # eigenvalues 1, 3
    res <- jit_eval(nv_eigen(nv_array(A, dtype = "f64")))
    expect_equal(as.numeric(as_array(res$values)), c(3, 1), tolerance = 1e-12)
  })

  it("nv_eigen(symmetric = FALSE) errors clearly", {
    A <- nv_array(matrix(c(1, 2, 3, 4), 2), dtype = "f64")
    expect_error(
      nv_eigen(A, symmetric = FALSE),
      "only supports.*symmetric = TRUE"
    )
  })

  it("nv_eigen(only.values = TRUE) errors clearly", {
    A <- nv_array(matrix(c(2, 1, 1, 2), 2), dtype = "f64")
    expect_error(nv_eigen(A, only.values = TRUE), "only.values")
  })
})
