test_that("error message when using different platforms", {
  skip_if(!is_cuda())
  f <- jit(\(x, y) x + y)
  x <- nv_array(1, device = "cpu")
  y <- nv_array(1, device = "cuda")
  expect_error(f(x, y), "Inputs live on different devices")
})

test_that("donate: must be formal args of f", {
  expect_error(jit(function(x) x, donate = "y"), "subset of")
})

test_that("donate: cannot also be static", {
  expect_error(jit(function(x, y) x, donate = "x", static = "x"), "donate.*static")
})

test_that("donate: no aliasing with type mismatch", {
  skip_if(!is_cpu()) # might get a segfault on other platforms
  f <- jit(function(x) x, device = "cpu", donate = "x")
  x <- nv_array(1)
  out <- f(x)
  expect_error(capture.output(x), "called on deleted or donated buffer")
})

test_that("jit: respects device argument as string", {
  f <- jit(function() 1, device = "cpu")
  expect_equal(f(), nv_scalar(1, device = "cpu", ambiguous = TRUE))
})

test_that("jit: respects device argument as PJRTDevice", {
  f <- jit(\() 1, device = pjrt::pjrt_device("cpu"))
  expect_equal(f(), nv_scalar(1, device = "cpu", ambiguous = TRUE))
})

test_that("from_arg explicitly wires the device into xla compilation with a renamed formal", {
  skip_if(!is_cpu())
  # The formal is renamed to `dev` (not `device`) to rule out the
  # formals-shadowing coincidence where the xla jit's closure `device`
  # happens to be shadowed by a same-named formal.
  captured <- NULL
  orig <- compile_to_xla
  local_mocked_bindings(
    compile_to_xla = function(f, args_flat, in_tree, donate = character(), device = NULL) {
      captured <<- device
      orig(f, args_flat, in_tree, donate = donate, device = device)
    }
  )

  f <- jit(
    function(shape, dev = NULL) nv_fill(1, shape = shape, dtype = "f32"),
    static = c("shape", "dev"),
    device = from_arg("dev")
  )
  dev <- pjrt::pjrt_device("cpu")
  out <- f(shape = 3L, dev = dev)

  expect_s3_class(out, "AnvilArray")
  expect_equal(as.character(captured), as.character(dev))
})

test_that("xla: basic test", {
  f_add <- function(x, y) x + y
  args <- list(x = nv_aval("f32", c()), y = nv_aval("f32", c()))
  f_compiled <- xla(f_add, args = args)
  result <- f_compiled(nv_scalar(1, dtype = "f32"), nv_scalar(2, dtype = "f32"))
  expect_equal(result, nv_scalar(3, dtype = "f32"))
})
