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
  )
  expect_equal(f(nv_tensor(1), TRUE), nv_tensor(2))
  expect_equal(f(nv_tensor(1), FALSE), nv_tensor(1))
})


test_that("jit: tensor return value is not wrapped in list", {
  f <- jit(nvl_add)
  out <- f(nv_scalar(1.2), nv_scalar(-0.7))
  expect_equal(as_array(out), 1.2 + (-0.7), tolerance = 1e-6)
})

test_that("error message when using wrong device", {
  skip_if(!is_metal())
  f <- jit(\(x) x, device = "cpu")
  x <- nv_tensor(1, device = "metal")
  expect_error(f(x), "but buffer has device metal")
})
