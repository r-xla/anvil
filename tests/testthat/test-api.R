test_that("auto-broadcasting higher-dimensional tensors is not supported (it's bug prone)", {
  x <- nv_tensor(1:2, shape = c(2, 1))
  y <- nv_tensor(1:2, shape = c(1, 2))
  expect_error(
    jit(nv_add)(x, y),
    "must have the same shape"
  )
})

test_that("broadcasting scalars", {
  fjit <- jit(nv_add)
  expect_equal(
    fjit(
      nv_scalar(1),
      nv_tensor(0, shape = c(2, 2))
    ),
    nv_tensor(1, shape = c(2, 2))
  )
})

test_that("infix add", {
  f <- jit(function(x, y) {
    x + y
  })
  expect_equal(
    f(
      nv_tensor(1, shape = c(2, 2)),
      nv_tensor(0, shape = c(2, 2))
    ),
    nv_tensor(1, shape = c(2, 2))
  )
})

test_that("jit constant single return is bare tensor", {
  f <- jit(function() nv_scalar(0.5))
  out <- f()
  expect_equal(as_array(out), 0.5, tolerance = 1e-6)
})

test_that("Summary group generics", {
  fsum <- jit(function(x) sum(x))
  expect_equal(as_array(fsum(nv_tensor(1:10))), 55)
})

test_that("mean", {
  fmean <- jit(function(x) mean(x))
  expect_equal(as_array(fmean(nv_tensor(1:10, "f32"))), 5.5)
})

test_that("constants can be lifted to the appropriate level", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_equal(
    jit(gradient(f, wrt = "x"))(nv_scalar(2))[[1L]],
    nv_scalar(1)
  )
})

test_that("wrt non-existent argument", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_error(
    jit(gradient(f, wrt = "y"))(nv_tensor(2)),
    "must be a subset"
  )
})

test_that("promote to common", {
  f <- function(x, y) {
    nv_add(x, y)
  }
  expect_equal(
    jit(f)(nv_tensor(1, dtype = "i32"), nv_tensor(1.0, dtype = "f32")),
    nv_tensor(2.0, dtype = "f32")
  )
})

describe("nv_concatenate", {
  it("auto-promotes to common", {
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(c(1, 2)), nv_tensor(3:4))),
      nv_tensor(c(1, 2, 3, 4))
    )
  })
  it("can concatenate literals", {
    expect_equal(
      jit_eval(nv_concatenate(1L, 2L)),
      nv_tensor(1:2)
    )
    expect_equal(
      jit_eval(nv_concatenate(1L, 2L, dimension = 1L)),
      nv_tensor(1:2)
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1:2), 3L)),
      nv_tensor(1:3)
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1L), 2L)),
      nv_tensor(1:2)
    )
  })
  it("fails when dimension is out of bounds", {
    expect_error(
      jit_eval(nv_concatenate(nv_tensor(1:2, shape = c(2, 1)), nv_tensor(3:4, shape = c(2, 1)), dimension = 3L))
    )
  })
  it("can concatenate 2d tensors", {
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1:2, shape = c(2, 1)), nv_tensor(3:4, shape = c(2, 1)), dimension = 1L)),
      nv_tensor(1:4, shape = c(4, 1))
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1:2, shape = c(2, 1)), nv_tensor(3:4, shape = c(2, 1)), dimension = 2L)),
      nv_tensor(1:4, shape = c(2, 2), dtype = "i32")
    )
  })
  it("fails with incompatible shapes", {
    expect_error(
      jit_eval(nv_concatenate(nv_tensor(1, shape = c(1, 1, 1)), nv_tensor(2, shape = c(1, 1)), dimension = 1L))
    )
  })
})
