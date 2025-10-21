expect_jit_unary <- function(nv_fun, rfun, x, scalar = !is.array(x)) {
  f <- jit(function(a) {
    nv_fun(a)
  })
  out <- if (scalar) {
    f(nv_scalar(x))
  } else {
    f(nv_tensor(x))
  }
  testthat::expect_equal(as_array(out), rfun(x), tolerance = 1e-6)
}

expect_jit_binary <- function(nv_fun, rfun, x, y, scalar = TRUE) {
  f <- jit(function(a, b) {
    nv_fun(a, b)
  })
  out <- if (scalar) {
    f(nv_scalar(x), nv_scalar(y))
  } else {
    f(nv_tensor(x), nv_tensor(y))
  }
  testthat::expect_equal(as_array(out), rfun(x, y), tolerance = 1e-6)
}

expect_grad_unary <- function(nv_fun, d_rfun, x) {
  gfun <- jit(gradient(function(a) nv_fun(a)))
  gout <- gfun(nv_scalar(x))[[1L]]
  testthat::expect_equal(as_array(gout), d_rfun(x), tolerance = 1e-5)
}

expect_grad_binary <- function(nv_fun, d_rx, d_ry, x, y) {
  gfun <- jit(gradient(function(a, b) nv_fun(a, b)))
  gout <- gfun(nv_scalar(x), nv_scalar(y))
  gx <- as_array(gout[[1L]])
  gy <- as_array(gout[[2L]])

  testthat::expect_equal(gx, d_rx(x, y), tolerance = 1e-5)
  testthat::expect_equal(gy, d_ry(x, y), tolerance = 1e-5)
}

nvj_add <- jit(nv_add)
nvj_mul <- jit(nv_mul)

skip_if_metal <- function(msg = "") {
  if (is_metal()) {
    testthat::skip(sprintf("Skipping test on Metal device: %s", msg))
  }
}

is_metal <- function() {
  Sys.getenv("PJRT_PLATFORM") == "metal"
}

is_cuda <- function() {
  Sys.getenv("PJRT_PLATFORM") == "cuda"
}

generate_test_data <- function(dimension, dtype = "f64", non_negative = FALSE) {
  if (dtype == "pred") {
    sample(c(TRUE, FALSE), size = prod(dimension), replace = TRUE)
  } else if (dtype %in% c("ui8", "ui16", "ui32", "ui64")) {
    sample(0:20, size = prod(dimension), replace = TRUE)
  } else if (dtype %in% c("i8", "i16", "i32", "i64")) {
    test_data <- as.integer(rgeom(prod(dimension), .5))
    if (!non_negative) {
      test_data <- as.integer((-1)^rbinom(prod(dimension), 1, .5) * test_data)
    }
    test_data
  } else {
    if (!non_negative) {
      rnorm(prod(dimension), mean = 0, sd = 1)
    } else {
      rchisq(prod(dimension), df = 1)
    }
  }
}

if (requireNamespace("torch", quietly = TRUE)) {
  source(system.file("extra-tests", "torch-helpers.R", package = "anvil"))
}
