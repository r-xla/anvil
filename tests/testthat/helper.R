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

skip_if_not_cpu <- function(msg = "") {
  if (is_cuda()) {
    testthat::skip(sprintf("Skipping test on %s device: %s", platform, msg))
  }
}

is_cuda <- function() {
  Sys.getenv("PJRT_PLATFORM") == "cuda"
}

is_cpu <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu") == "cpu"
}

generate_test_data <- function(dimension, dtype = "f64", non_negative = FALSE) {
  data <- if (dtype == "pred") {
    sample(c(TRUE, FALSE), size = prod(dimension), replace = TRUE)
  } else if (dtype %in% c("ui8", "ui16", "ui32", "ui64")) {
    sample(0:20, size = prod(dimension), replace = TRUE)
  } else if (dtype %in% c("i8", "i16", "i32", "i64")) {
    test_data <- as.integer(rgeom(prod(dimension), 0.5))
    if (!non_negative) {
      test_data <- as.integer((-1)^rbinom(prod(dimension), 1, 0.5) * test_data)
    }
    test_data
  } else {
    if (!non_negative) {
      rnorm(prod(dimension), mean = 0, sd = 1)
    } else {
      rchisq(prod(dimension), df = 1)
    }
  }

  array(data, dim = dimension)
}

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "torch-helpers.R", package = "anvil"))
}


verify_zero_grad_unary <- function(nvl_fn, x, f_wrapper = NULL) {
  if (is.null(f_wrapper)) {
    f <- function(x) {
      out <- nvl_fn(x)
      out <- nv_convert(out, "f32")
      nv_reduce_sum(out, dims = 1L, drop = TRUE)
    }
  } else {
    f <- f_wrapper
  }
  grads <- jit(gradient(f))(x)
  shp <- shape(x)
  expected <- nv_tensor(0, shape = shp, dtype = dtype(x), ambiguous = ambiguous(x))
  testthat::expect_equal(grads[[1L]], expected)
}

verify_zero_grad_binary <- function(nvl_fn, x, y) {
  f <- function(x, y) {
    out <- nvl_fn(x, y)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }
  grads <- jit(gradient(f))(x, y)
  shp <- shape(x)
  expected1 <- nv_tensor(0, shape = shp, dtype = dtype(x), ambiguous = ambiguous(x))
  expected2 <- nv_tensor(0, shape = shp, dtype = dtype(y), ambiguous = ambiguous(y))
  testthat::expect_equal(grads[[1L]], expected1)
  testthat::expect_equal(grads[[2L]], expected2)
}
