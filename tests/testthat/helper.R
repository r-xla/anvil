expect_jit_equal <- function(.expr, .expected, ...) {
  expr <- substitute(.expr)
  eval_env <- new.env(parent = parent.frame())
  observed <- jit(\() eval(expr, envir = eval_env))()
  testthat::expect_equal(observed, .expected, ...)
}

# Run an API function twice in eager mode -- once with every arrayish input
# placed on cpu:0 and once on cpu:1 -- and assert the two outputs are equal.
# Non-arrayish arguments are passed through unchanged.
check_eager <- function(fn, ..., tolerance = 1e-6) {
  dev0 <- nv_device("cpu:0", "xla")
  dev1 <- nv_device("cpu:1", "xla")
  args <- list(...)

  place_on <- function(x, dev) {
    if (is_anvil_array(x)) {
      nv_array(
        as_array(x),
        dtype = dtype(x),
        device = dev,
        shape = shape(x),
        ambiguous = ambiguous(x)
      )
    } else if (is.numeric(x) || is.logical(x)) {
      nv_array(x, device = dev)
    } else {
      x
    }
  }

  to_r <- function(x) {
    if (is_anvil_array(x)) {
      as_array(x)
    } else if (is.list(x)) {
      lapply(x, to_r)
    } else {
      x
    }
  }

  # Recursively walk `out` and assert every AnvilArray lives on `dev`.
  check_on_device <- function(out, dev) {
    if (is_anvil_array(out)) {
      testthat::expect_true(
        eq_device(device(out), dev),
        info = sprintf(
          "expected output on %s, got %s",
          as.character(dev),
          as.character(device(out))
        )
      )
    } else if (is.list(out)) {
      lapply(out, check_on_device, dev = dev)
    }
    invisible(NULL)
  }

  out0 <- do.call(fn, lapply(args, place_on, dev = dev0))
  out1 <- do.call(fn, lapply(args, place_on, dev = dev1))
  check_on_device(out0, dev0)
  check_on_device(out1, dev1)
  testthat::expect_equal(to_r(out0), to_r(out1), tolerance = tolerance)
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
