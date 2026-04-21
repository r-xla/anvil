expect_jit_equal <- function(.expr, .expected, ...) {
  expr <- substitute(.expr)
  eval_env <- new.env(parent = parent.frame())
  observed <- jit(\() eval(expr, envir = eval_env))()
  testthat::expect_equal(observed, .expected, ...)
}

# Cross-check an API function by running it in two configurations:
#   - eager mode with every AnvilArray input on cpu:1
#   - jit-compiled (static args forwarded) with every AnvilArray input on cpu:0
# and asserting that
#   1. the eager outputs live on cpu:1
#   2. the jit outputs live on cpu:0
#   3. the two outputs agree value-wise (as plain R structures)
#
# `static` is forwarded to `jit()` and names the arguments that the jitted
# compilation should capture as compile-time constants.
check_eager <- function(fn, ..., static = character(), tolerance = 1e-6) {
  dev_eager <- nv_device("cpu:1", "xla")
  dev_jit <- nv_device("cpu:0", "xla")
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

  out_eager <- do.call(fn, lapply(args, place_on, dev = dev_eager))
  out_jit <- do.call(
    jit(fn, static = static),
    lapply(args, place_on, dev = dev_jit)
  )
  check_on_device(out_eager, dev_eager)
  check_on_device(out_jit, dev_jit)
  testthat::expect_equal(to_r(out_eager), to_r(out_jit), tolerance = tolerance)
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
