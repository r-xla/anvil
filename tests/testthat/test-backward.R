test_that("basic pullback test", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  f_grad <- jit(gradient(f))
  out <- f_grad(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("simple function works (scalar)", {
  f_grad <- jit(gradient(nvl_mul))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(2.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("chain rule works (scalar)", {
  f <- function(x, y) {
    nvl_add(nvl_mul(x, y), x)
  }

  f_grad <- jit(gradient(f))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(3.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("gradient does not have to depend on input", {
  # This is special, because the input has no influence
  # on the gradient, because the gradient is constant
  f <- function(x, y) {
    nvl_add(x, y)
  }

  f_grad <- jit(gradient(f))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("nested inputs", {
  f <- jit(gradient(function(x) {
    nvl_mul(x[[1]][[1]], x[[1]][[1]])
  }))
  expect_equal(
    f(list(list(nv_scalar(1)))),
    list(x = list(list(nv_scalar(2))))
  )
})

test_that("no nested outpus", {
  # we expect a scalar output
  # -> check for good error message
})

test_that("constants work (scalar)", {
  f <- jit(gradient(function(x) {
    nvl_mul(x, nv_scalar(2))
  }))
  expect_equal(f(nv_scalar(1)), list(x = nv_scalar(2.0)))
})

test_that("broadcasting works", {
  # TODO
})

test_that("second order gradient (scalar)", {
  # this works only for scalar functions, so this is primarily a stress
  # test for out transformation implementation, not because it's useful in itself.
  f <- function(x) {
    nvl_mul(x, x)
  }
  fg2 <- jit(gradient(\(x) gradient(f)(x)[[1L]]))
  expect_equal(
    fg2(nv_scalar(1)),
    list(x = nv_scalar(2.0))
  )
})

test_that("neg works", {
  g <- jit(gradient(nvl_negate))
  expect_equal(g(nv_scalar(1))[[1L]], nv_scalar(-1))
})

test_that("names for grad: primitive", {
  g <- jit(gradient(`*`))
  expect_equal(formalArgs2(g), c("e1", "e2"))
  expect_equal(
    g(nv_scalar(2), nv_scalar(1)),
    list(e1 = nv_scalar(1.0), e2 = nv_scalar(2.0))
  )
})

test_that("names for grad: function", {
  f <- function(e1, e2) {
    e1 * e2
  }
  g <- jit(gradient(f))
  expect_equal(formals(g), formals(f))
  result <- g(nv_scalar(2), nv_scalar(1))
  expect_equal(result, list(e1 = nv_scalar(1.0), e2 = nv_scalar(2.0)))
})

# New tests for selective gradients (wrt)

test_that("partial gradient simple", {
  f <- function(lhs, rhs) {
    nvl_add(lhs, rhs)
  }
  g <- jit(gradient(f, wrt = "lhs"))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))[[1L]]
  expect_equal(out, nv_scalar(1.0))
})

#test_that("partial gradient: y = a * (x * b) wrt x", {
#  f <- function(a, x, b) {
#    a * (x * b)
#  }
#  g <- gradient(f, wrt = "x")
#  a <- nv_scalar(2.0)
#  x <- nv_scalar(3.0)
#  b <- nv_scalar(5.0)
#  out <- g(a, x, b)
#  expect_null(out[[1L]])
#  expect_equal(out[[2L]], nv_scalar(10.0))
#  expect_null(out[[3L]])
#})
#

#test_that("pullback", {
#  fbwd <- jit(pullback(nv_add, lhs = nv_scalar(1.0), rhs = nv_scalar(2.0), wrt = "lhs"))
#  expect_equal(fbwd(nv_scalar(10.0)), list(lhs = nv_scalar(10.0)))
#})
#
#test_that("pullback: non-scalar", {
#  fbwd <- jit(pullback(nv_mul, lhs = nv_tensor(1:10), rhs = nv_tensor(2:11), wrt = "lhs"))
#  x <- nv_tensor(1:10)
#  expect_equal(fbwd(x), list(lhs = jit(nv_mul)(x, nv_tensor(2:11))))
#})

test_that("gradients are present even if they don't influence the output", {
  g <- jit(gradient(function(x, y) x, wrt = "y"))
  expect_equal(
    g(nv_scalar(1), nv_scalar(1)),
    list(y = nv_scalar(0))
  )

  g2 <- jit(gradient(function(x, y) {
    z <- nv_mul(x, x)
    return(y)
  }))
  expect_equal(
    g2(nv_scalar(1), nv_scalar(1)),
    list(x = nv_scalar(0.0), y = nv_scalar(1.0))
  )
})

test_that("wrt non-existent argument", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_error(
    jit(gradient(f, wrt = "y"))(nv_scalar(2)),
    "wrt must be"
  )
})

test_that("gradient: simple example", {
  f <- function(x, y) {
    nvl_mul(x, y)
  }
  g <- jit(gradient(f))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(2.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("gradient: does not depend on input", {
  f <- function(x, y) {
    nvl_add(x, y)
  }
  g <- jit(gradient(f))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("wrt for non-tensor input: gradient", {
  expect_snapshot(error = TRUE, {
    g <- gradient(nv_round, wrt = "method")
    g(nv_scalar(1), method = "nearest_even")
  })
})

test_that("wrt for non-tensor input: value_and_gradient", {
  expect_snapshot(error = TRUE, {
    g <- value_and_gradient(nv_round, wrt = "method")
    g(nv_scalar(1), method = "nearest_even")
  })
})

test_that("wrt for nested non-tensor input: gradient", {
  f <- function(x) {
    nvl_mul(x[[1]], x[[2]])
  }
  expect_snapshot(error = TRUE, {
    g <- gradient(f, wrt = "x")
    g(x = list(nv_scalar(1), 2L))
  })
})

test_that("wrt for nested non-tensor input: value_and_gradient", {
  f <- function(x) {
    nvl_mul(x[[1]], x[[2]])
  }
  expect_snapshot(error = TRUE, {
    g <- value_and_gradient(f, wrt = "x")
    g(x = list(nv_scalar(1), 2L))
  })
})

test_that("can only compute gradient w.r.t. float tensors", {
  expect_snapshot(error = TRUE, {
    gradient(nv_floor, wrt = "operand")(nv_scalar(1L))
  })
})

test_that("can differentiate through integer/bool functions", {
  f <- function(x) {
    x1 <- nv_convert(x, "i32")
    x2 <- nvl_popcnt(x1)
    x3 <- nv_convert(x2, "f32")
    mean(x3)
  }
  g <- jit(gradient(f))
  expect_equal(
    g(nv_tensor(c(1, 2))),
    list(x = nv_tensor(c(0, 0)))
  )
})

test_that("gradient with static (non-tensor) argument", {
  f <- function(x, y) {
    if (x) y * y else y * 7
  }
  g <- jit(gradient(f, wrt = "y"), static = "x")

  # x=TRUE -> y*y -> dy/dy = 2*y = 6
  out_true <- g(TRUE, nv_scalar(3.0))
  expect_equal(out_true[[1L]], nv_scalar(6.0))

  # x=FALSE -> y*7 -> dy/dy = 7
  out_false <- g(FALSE, nv_scalar(3.0))
  expect_equal(out_false[[1L]], nv_scalar(7.0))
})

test_that("value_and_gradient with static (non-tensor) argument", {
  f <- function(x, y) {
    if (x) y * y else y * 7
  }
  vg <- jit(value_and_gradient(f, wrt = "y"), static = "x")

  # x=TRUE -> y*y = 9, dy/dy = 6
  result_true <- vg(TRUE, nv_scalar(3.0))
  expect_equal(result_true$value, nv_scalar(9.0))
  expect_equal(result_true$grad[[1L]], nv_scalar(6.0))

  # x=FALSE -> y*7 = 21, dy/dy = 7
  result_false <- vg(FALSE, nv_scalar(3.0))
  expect_equal(result_false$value, nv_scalar(21.0))
  expect_equal(result_false$grad[[1L]], nv_scalar(7.0))
})

test_that("Can propagate ambiguous float32 through integer/bool functions", {
  f <- function(x) {
    x1 <- nv_convert(x, "i32")
    x2 <- nv_convert(x1, "i1")
    x3 <- nvl_not(x1)
    x4 <- nv_convert(x3, "f32")
    mean(x4)
  }
  grad <- jit(gradient(f))
  grad(nv_scalar(1))
})

test_that("trace_fn matches args with formals", {
  graph1 <- trace_fn(nvl_add, list(nv_aten("f32", c()), nv_aten("f32", c())))
  graph2 <- trace_fn(nvl_add, list(lhs = nv_aten("f32", c()), rhs = nv_aten("f32", c())))
  expect_equal(graph1$in_tree, graph2$in_tree)
})
