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
})

test_that("calling jit on jit", {
  # TODO: (Not sure what we want here)
})
