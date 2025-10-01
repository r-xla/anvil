test_that("broadcasting", {
  fjit <- jit(nv_add)
  expect_equal(
    fjit(
      nv_tensor(1),
      nv_tensor(0, shape = c(2, 2))
    ),
    nv_tensor(1, shape = c(2, 2))
  )
})

test_that("infix add", {
  f <- jit(function(x, y) {
    x + y
  })
  f(
    nv_tensor(1),
    nv_tensor(0, shape = c(2, 2))
  )
  expect_equal(
    f(
      nv_tensor(1),
      nv_tensor(0, shape = c(2, 2))
    ),
    nv_tensor(1, shape = c(2, 2))
  )
})

test_that("jit single return without list wrapper", {
  f <- jit(nvl_add)
  out <- f(nv_scalar(1.2), nv_scalar(-0.7))
  expect_equal(as_array(out), 1.2 + (-0.7), tolerance = 1e-6)
})

test_that("jit constant single return is bare tensor", {
  f <- jit(function() nv_scalar(0.5))
  out <- f()
  expect_equal(as_array(out), 0.5, tolerance = 1e-6)
})
