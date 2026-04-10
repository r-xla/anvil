test_that("jit: quickr backend compiles simple function", {
  skip_if_not_installed("quickr")
  local_backend("quickr")

  f <- jit(function(x, y) x + y)

  expect_equal(as_array(f(nv_scalar(1L), nv_scalar(2L))), array(3L, dim = 1L))
})

test_that("jit: quickr backend returns AnvilArray", {
  skip_if_not_installed("quickr")
  local_backend("quickr")

  f <- jit(function(x, y) x + y)
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(4, 5, 6))

  result <- f(x, y)
  expect_s3_class(result, "AnvilArray")
  expect_equal(backend(result), "quickr")
  expect_equal(as_array(result), array(c(5, 7, 9), dim = 3L))
})

test_that("jit: quickr backend preserves nested multi-output shapes and types", {
  skip_if_not_installed("quickr")
  local_backend("quickr")

  f <- jit(
    function(x) {
      list(
        flags = x > 1L,
        payload = list(shifted = x + 1L)
      )
    }
  )

  out <- f(nv_array(1:3))

  expect_identical(as.character(dtype(out$flags)), "bool")
  expect_identical(as.character(dtype(out$payload$shifted)), "i32")
  expect_identical(shape(out$flags), 3L)
  expect_identical(shape(out$payload$shifted), 3L)
  expect_identical(as_array(out$flags), array(c(FALSE, TRUE, TRUE), dim = 3L))
  expect_identical(as_array(out$payload$shifted), array(2:4, dim = 3L))
})

test_that("jit: quickr backend does not support donate or device", {
  local_backend("quickr")

  expect_error(
    jit(function(x) x, donate = "x"),
    "donate",
    fixed = TRUE
  )
  expect_error(
    jit(function(x) x, device = "cpu"),
    "device",
    fixed = TRUE
  )
})

test_that("jit: quickr backend traces floating literals as f32", {
  graph <- trace_fn(
    function() 1.0,
    list(),
    desc = local_descriptor()
  )

  expect_equal(dtype(graph$outputs[[1L]]$aval), as_dtype("f32"))
})

test_that("graph_to_r_function lowers a graph to a plain R function", {
  skip_if_not_installed("quickr")
  local_backend("quickr")

  graph <- trace_fn(
    function(x) x + 1L,
    list(x = nv_scalar(1.0, dtype = "f64")),
    desc = local_descriptor()
  )

  f <- graph_to_r_function(graph)

  expect_equal(f(2), 3)
})
