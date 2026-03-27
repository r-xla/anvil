expect_jit_equal <- function(.expr, .expected, ...) {
  expr <- substitute(.expr)
  eval_env <- new.env(parent = parent.frame())
  observed <- jit(\() eval(expr, envir = eval_env))()
  testthat::expect_equal(observed, .expected, ...)
}

is_cuda <- function() {
  Sys.getenv("PJRT_PLATFORM") == "cuda"
}

is_cpu <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu") == "cpu"
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
  expected <- nv_array(0, shape = shp, dtype = dtype(x), ambiguous = ambiguous(x))
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
  expected1 <- nv_array(0, shape = shp, dtype = dtype(x), ambiguous = ambiguous(x))
  expected2 <- nv_array(0, shape = shp, dtype = dtype(y), ambiguous = ambiguous(y))
  testthat::expect_equal(grads[[1L]], expected1)
  testthat::expect_equal(grads[[2L]], expected2)
}
