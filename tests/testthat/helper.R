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
  # We can only take gradients w.r.t. float arrays, so the outer input is f32
  # and the actual dtype is restored inside the function before calling nvl_fn.
  x_dtype <- dtype(x)
  x_f32 <- jit(nv_convert, static = "dtype")(x, "f32")
  if (is.null(f_wrapper)) {
    f <- function(x) {
      x_inner <- nv_convert(x, x_dtype)
      out <- nvl_fn(x_inner)
      out <- nv_convert(out, "f32")
      nv_reduce_sum(out, dims = 1L, drop = TRUE)
    }
  } else {
    f <- f_wrapper
  }
  grads <- jit(gradient(f))(x_f32)
  expected <- nv_array(0, shape = shape(x), dtype = "f32")
  testthat::expect_equal(grads[[1L]], expected)
}

verify_zero_grad_binary <- function(nvl_fn, x, y) {
  x_dtype <- dtype(x)
  y_dtype <- dtype(y)
  x_f32 <- jit(nv_convert, static = "dtype")(x, "f32")
  y_f32 <- jit(nv_convert, static = "dtype")(y, "f32")
  f <- function(x, y) {
    x_inner <- nv_convert(x, x_dtype)
    y_inner <- nv_convert(y, y_dtype)
    out <- nvl_fn(x_inner, y_inner)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }
  grads <- jit(gradient(f))(x_f32, y_f32)
  expected1 <- nv_array(0, shape = shape(x), dtype = "f32")
  expected2 <- nv_array(0, shape = shape(y), dtype = "f32")
  testthat::expect_equal(grads[[1L]], expected1)
  testthat::expect_equal(grads[[2L]], expected2)
}
