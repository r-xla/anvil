test_that("jit: basic test", {
  f <- jit(function(x, y) {
    nvl_add(x, y)
  })

  expect_equal(
    f(nv_scalar(1), nv_scalar(2)),
    nv_scalar(3)
  )
  expect_equal(
    f(nv_scalar(1), nv_scalar(2)),
    nv_scalar(3)
  )
})

test_that("jit: a constant", {
  x <- nv_scalar(1)
  f <- function(y) {
    nvl_add(x, y)
  }
  f_jit <- jit(f)
  expect_equal(
    f_jit(nv_scalar(2)),
    nv_scalar(3)
  )
  x <- nv_scalar(2)
  # the constant is now saved in f_jit, so new x is not found
  cache_size(f_jit)
  expect_equal(
    f_jit(nv_scalar(2)),
    nv_scalar(3)
  )
  cache_size(f_jit)
})

test_that("jit basic test", {
  f <- function(x, y) {
    x + y
  }
  f_jit <- jit(f)

  expect_equal(
    f_jit(nv_tensor(1), nv_tensor(2)),
    nv_tensor(3)
  )
  # cache hit:
  expect_equal(
    f_jit(nv_tensor(1), nv_tensor(2)),
    nv_tensor(3)
  )
})

test_that("can return single tensor", {
  f <- jit(nvl_add)
  expect_equal(
    f(
      nv_tensor(1.0),
      nv_tensor(-2.0)
    ),
    nv_tensor(-1.0)
  )
})

test_that("can return nested list", {
  f <- jit(\(x, y) {
    list(x, list(a = y))
  })
  x <- nv_tensor(1)
  y <- nv_tensor(2)
  expect_equal(
    f(x, y),
    list(x, list(a = y))
  )
})

test_that("can take in nested list", {
  f <- jit(\(x) {
    x$a
  })
  x <- nv_tensor(1)
  expect_equal(
    f(list(a = x)),
    x
  )
})

test_that("Can multiply values from lists", {
  f <- jit(function(x, y) {
    x[[1]] * y[[1]]
  })
  expect_equal(
    f(list(nv_scalar(2)), list(nv_scalar(3))),
    nv_scalar(6)
  )
})

test_that("multiple returns", {
  f <- jit(function(x, y) {
    list(
      nvl_add(x, y),
      nvl_mul(x, y)
    )
  })

  out <- f(
    nv_tensor(1.0),
    nv_tensor(2.0)
  )
  expect_equal(
    out[[1]],
    nv_tensor(3.0)
  )
  expect_equal(
    out[[2]],
    nv_tensor(2.0)
  )
})

test_that("jitted function has class JitFunction", {
  f_jit <- jit(function(x) x)
  expect_s3_class(f_jit, "JitFunction")
})

test_that("calling jit on jit errors", {
  f_jit <- jit(function(x) x)
  expect_error(jit(f_jit)(nv_tensor(1)), "must not be a jitted function")
})

test_that("keeps argument names", {
  f1 <- function(x, y) {
    x + y
  }
  f1_jit <- jit(f1)
  expect_equal(
    formals(f1_jit),
    formals(f1)
  )
  f1_jit(nv_tensor(1), nv_tensor(2))

  expect_equal(
    f1_jit(nv_tensor(1), nv_tensor(2)),
    nv_tensor(3)
  )
  f2 <- function(x, y = nv_tensor(1)) {
    x + y
  }
  f2_jit <- jit(f2)
  expect_equal(
    formals(f2_jit),
    formals(f2)
  )
  expect_equal(
    f2_jit(nv_tensor(1), nv_tensor(2)),
    nv_tensor(3)
  )
  f3 <- function(a, ...) {
    Reduce(`+`, list(...)) * a
  }
  f3_jit <- jit(f3)
  expect_equal(
    formals(f3_jit),
    formals(f3)
  )
  expect_equal(
    f3_jit(nv_tensor(2), nv_tensor(3), nv_tensor(4)),
    nv_tensor(14)
  )
})

test_that("can mark arguments as static ", {
  f <- jit(
    function(x, add_one) {
      if (add_one) {
        x + nv_tensor(1)
      } else {
        x
      }
    },
    static = "add_one"
    # TODO: Better error message ...
  )
  expect_equal(f(nv_tensor(1), TRUE), nv_tensor(2))
  expect_equal(f(nv_tensor(1), FALSE), nv_tensor(1))
})


test_that("jit: tensor return value is not wrapped in list", {
  f <- jit(nvl_add)
  out <- f(nv_scalar(1.2), nv_scalar(-0.7))
  expect_equal(as_array(out), 1.2 + (-0.7), tolerance = 1e-6)
})

test_that("error message when using different platforms", {
  skip_if(!is_cuda())
  f <- jit(\(x, y) x + y)
  x <- nv_tensor(1, device = "cpu")
  y <- nv_tensor(1, device = "cuda")
  expect_error(f(x, y), "Inputs live on different platforms")
})

test_that("constants can be part of the program", {
  f <- jit(function(x) x + nv_scalar(1))
  expect_equal(f(nv_tensor(1)), nv_tensor(2))
})

test_that("Only constants in group generics", {
  f <- jit(function() {
    nv_scalar(1) + nv_scalar(2)
    #nv_add(nv_scalar(1), nv_scalar(2))
  })
  expect_equal(f(), nv_scalar(3))
})

test_that("donate: no aliasing with type mismatch", {
  skip_if(!is_cpu()) # might get a segfault on other platforms
  f <- jit(function(x) x, donate = "x")
  x <- nv_tensor(1)
  out <- f(x)
  expect_error(capture.output(x), "called on deleted or donated buffer")
})

test_that("... works (#19)", {
  expect_equal(
    jit(sum)(nv_tensor(1:10)),
    nv_scalar(55L, dtype = "i32")
  )

  f <- function(..., a) {
    a + sum(...)
  }
  expect_equal(
    jit(f)(a = nv_scalar(1L), nv_tensor(1:10)),
    nv_scalar(56L, dtype = "i32")
  )
})

test_that("good error message when passing AbstractTensors", {
  expect_error(jit(nv_negate)(nv_aten("f32", c(2, 2))), "Expected AnvilTensor")
})

test_that("jit: respects device argument", {
  f <- jit(function() 1, device = "cpu")
  expect_equal(f(), nv_scalar(1, device = "cpu"))
})

test_that("literals are not converted to scalar tensors", {
  f <- jit(nv_sine)
  expect_error(f(1), "Expected AnvilTensor")
})

test_that("jit_eval does not modify calling environment", {
  x <- nv_tensor(1:2)
  jit_eval({
    x <- nv_tensor(3:4)
  })
  expect_equal(x, nv_tensor(1:2))
})

test_that("xla: basic test", {
  f_add <- function(x, y) x + y
  args <- list(x = nv_aten("f32", c()), y = nv_aten("f32", c()))
  f_compiled <- xla(f_add, args = args)
  result <- f_compiled(nv_scalar(1, dtype = "f32"), nv_scalar(2, dtype = "f32"))
  expect_equal(result, nv_scalar(3, dtype = "f32"))
})

test_that("hash for cache depends on in_tree (#122)", {
  f <- jit(\(...) {
    args <- list(...)
    args[[1]][[1L]][[1L]]
  })
  expect_equal(cache_size(f), 0L)
  expect_equal(f(list(list(nv_scalar(1L)), nv_scalar(2L))), nv_scalar(1L))
  expect_equal(cache_size(f), 1L)
  expect_equal(f(list(list(nv_scalar(1L), nv_scalar(2L)))), nv_scalar(1L))
  expect_equal(cache_size(f), 2L)
})
