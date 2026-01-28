test_that("ambiguity is propagated by binary ops", {
  f <- function(x, y) {
    (x * 1L) + y
  }
  expect_equal(
    jit(f)(nv_scalar(TRUE), nv_scalar(2L, "i16")),
    nv_scalar(3L, dtype = "i16")
  )
})

test_that("ambiguity is propagated by unary ops", {
  f <- function(x) {
    nv_negate(1L) + x
  }
  expect_equal(
    jit(f)(nv_scalar(2L, "i16")),
    nv_scalar(1L, dtype = "i16")
  )
})

test_that("p_convert backward", {
  out <- jit(function(x) {
    z <- x * 1L
    a <- gradient(\(y) {
      nvl_convert(y, "f32", ambiguous = TRUE)
    })(z)[[1L]] *
      nv_scalar(1, dtype = "i16")
  })(nv_scalar(TRUE))
  expect_equal(out, nv_scalar(1L, dtype = "i16"))
})

test_that("p_if propagates ambiguity", {
  f <- function(pred, x) {
    x <- x * 2L
    nv_if(pred, x, x * x) * nv_scalar(3L, dtype = "i16")
  }
  expect_equal(
    jit(f)(nv_scalar(TRUE), nv_scalar(TRUE)),
    nv_scalar(6L, dtype = "i16")
  )
})

test_that("p_while propagates ambiguity", {
  f <- jit(function(n) {
    i <- 0L * nv_scalar(TRUE) # ambiguous i32
    nv_while(list(i = i), \(i) i <= n, \(i) {
      i <- i + 1L
      list(i = i)
    })[[1L]] *
      nv_scalar(3L, dtype = "i16")
  })
  expect_equal(
    f(nv_scalar(10L)),
    nv_scalar(33L, dtype = "i16")
  )
})

test_that("boolean is not ambiguous", {
  f <- function(x) {
    x * TRUE
  }
  graph <- trace_fn(f, list(x = nv_scalar(1L)))
  expect_false(graph$calls[[1L]]$inputs[[1L]]$aval$ambiguous)
})
