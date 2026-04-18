test_that("error message when using different platforms", {
  skip_if(!is_cuda())
  f <- jit(\(x, y) x + y)
  x <- nv_array(1, device = "cpu")
  y <- nv_array(1, device = "cuda")
  expect_error(f(x, y), "on unexpected device")
})

test_that("donate: must be formal args of f", {
  expect_error(jit(function(x) x, donate = "y"), "subset of")
})

test_that("donate: cannot also be static", {
  expect_error(jit(function(x, y) x, donate = "x", static = "x"), "donate.*static")
})

test_that("donate: no aliasing with type mismatch", {
  skip_if(!is_cpu()) # might get a segfault on other platforms
  f <- jit(function(x) x, donate = "x")
  x <- nv_array(1)
  out <- f(x)
  expect_error(capture.output(x), "called on deleted or donated buffer")
})

test_that("xla: basic test", {
  f_add <- function(x, y) x + y
  args <- list(x = nv_aval("f32", c()), y = nv_aval("f32", c()))
  f_compiled <- xla(f_add, args = args)
  result <- f_compiled(nv_scalar(1, dtype = "f32"), nv_scalar(2, dtype = "f32"))
  expect_equal(result, nv_scalar(3, dtype = "f32"))
})
