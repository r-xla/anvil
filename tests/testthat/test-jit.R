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
  # the constant is now saved in f_jit, so new x is not foundj
  expect_equal(
    f_jit(nv_scalar(2)),
    nv_scalar(3)
  )
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

test_that("calling jit on jit", {
  # TODO: (Not sure what we want here)
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
    nv_scalar(55L, dtype = "i32") # explicit dtype to match non-ambiguous input
  )

  f <- function(..., a) {
    a + sum(...)
  }
  expect_equal(
    jit(f)(a = nv_scalar(1L), nv_tensor(1:10)),
    nv_scalar(56L, dtype = "i32") # explicit dtype to match non-ambiguous result
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

test_that("jit_eval modifies parent frame variables", {
  x <- nv_scalar(1)
  jit_eval(x <- x + nv_scalar(1))
  expect_equal(x, nv_scalar(2))
})

test_that("ambiguous and non-ambiguous inputs generate different executables", {
  # Use an environment to track trace count without using <<-
  counter <- new.env()
  counter$trace <- 0

  f <- jit(function(x) {
    counter$trace <- counter$trace + 1
    x + nv_scalar(counter$trace)
  })

  # First call with ambiguous input (f32)
  result1 <- f(nv_scalar(10.0))
  expect_equal(result1, nv_scalar(11.0))
  expect_equal(counter$trace, 1)

  # Second call with same ambiguous input (cache hit)
  result2 <- f(nv_scalar(20.0))
  expect_equal(result2, nv_scalar(21.0))
  expect_equal(counter$trace, 1) # No recompilation

  # Third call with non-ambiguous input of same dtype (cache miss due to different ambiguity)
  result3 <- f(nv_tensor(10.0, dtype = "f32"))
  expect_equal(result3, nv_tensor(12.0, dtype = "f32")) # counter$trace is now 2
  expect_equal(counter$trace, 2)

  # Fourth call with same non-ambiguous input (cache hit)
  result4 <- f(nv_tensor(20.0, dtype = "f32"))
  expect_equal(result4, nv_tensor(22.0, dtype = "f32"))
  expect_equal(counter$trace, 2) # No recompilation

  # Fifth call with ambiguous i32 input (cache miss - different dtype)
  result5 <- f(nv_scalar(10L))
  expect_equal(result5, nv_scalar(13, dtype = "f32", ambiguous = TRUE))
  expect_equal(counter$trace, 3)
})
