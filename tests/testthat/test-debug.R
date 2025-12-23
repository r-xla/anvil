test_that("create debug box", {
  box <- debug_box("f32", c(2, 2))
  expect_equal(box@aval, AbstractTensor("f32", c(2, 2), FALSE))
})

test_that("printer for debug box", {
  expect_snapshot(
    DebugBox(AbstractTensor("f32", c(2, 2), TRUE))
  )
  expect_snapshot(
    DebugBox(AbstractTensor("f32", c(2, 2), FALSE))
  )
  expect_snapshot(
    DebugBox(AbstractTensor("f32", c(), FALSE))
  )
  expect_snapshot(
    DebugBox(ConcreteTensor(nv_tensor(1:4, dtype = "f32", shape = c(2, 2))))
  )
  expect_snapshot(
    DebugBox(LiteralTensor(1, shape = c(2, 3), ambiguous = TRUE))
  )
  expect_snapshot(
    DebugBox(LiteralTensor(1, shape = c(2, 3), ambiguous = FALSE))
  )
  expect_snapshot(
    DebugBox(LiteralTensor(1, shape = c(), ambiguous = FALSE))
  )
})

test_that("basic primitive", {
  # with debug box
  box <- debug_box("f32", c(2, 2))
  out <- nv_add(box, box)
  expect_class(out, "anvil::DebugBox")
  expect_equal(out@aval, box@aval)

  # with tensor
  x <- nv_tensor(1:4, dtype = "f32", shape = c(2, 2))
  out <- nv_add(x, x)
  expect_class(out, "anvil::DebugBox")
  expect_equal(out@aval, nv_aten("f32", c(2, 2)))
})

test_that("gradient", {
  f <- function(x) {
    mean(x * x)
  }
  ain <- debug_box("f32", c(2, 2))
  out <- gradient(f)(ain)[[1L]]
  expect_class(out, "anvil::DebugBox")
  expect_equal(out@aval, ain@aval)
})

test_that("can debug value_and_gradient", {
  f <- function(x) {
    mean(nv_add(x, x))
  }
  ain <- debug_box("f32", 2)
  g <- value_and_gradient(f)
  g(ain)
  out <- g(nv_tensor(1:2, dtype = "f32"))
  # In debug mode, returns AbstractTensors
  expect_equal(shape(out$value), integer())
  #expect_equal(shape(out$grad$x), 2L)
})

test_that("while", {
  f <- function(x) {
    nv_while(list(x = x), \(x) nv_gt(x, 0), \(x) {
      x <- nv_sub(x, 1)
      list(x = x)
    })
  }
  ain <- debug_box("f32", integer())
  f(ain)
})

test_that("if", {
  f <- function(x) {
    nv_if(nv_scalar(TRUE), x, x)
  }
  ain <- debug_box("f32", 1)
  expect_equal(f(ain)@aval, ain@aval)
})

test_that("group generics", {
  ain <- debug_box("f32", c(2, 2))
  scalar_aval <- AbstractTensor("f32", integer(), FALSE)

  # Ops: arithmetic
  expect_equal((ain + ain)@aval, ain@aval)
  # Ops: comparison
  expect_equal((ain > ain)@aval, AbstractTensor("pred", c(2, 2), FALSE))
  # matrixOps
  expect_equal((ain %*% ain)@aval, ain@aval)
  # Math
  expect_equal(exp(ain)@aval, ain@aval)
  # Math2
  expect_equal(round(ain)@aval, ain@aval)
  # Summary
  expect_equal(sum(ain)@aval, scalar_aval)
  # mean
  expect_equal(mean(ain)@aval, scalar_aval)
  # transpose
  ain2 <- debug_box("f32", c(2, 3))
  expect_equal(t(ain2)@aval, AbstractTensor("f32", c(3, 2), FALSE))
})


test_that("can't debug with abstract tensors", {
  # We can't allow this, because it might be confusing, because `==` does not call into
  # nvl_eq for AbstractTensors, but actually checks for type equality
  expect_error(
    nv_add(nv_aten("f32", c(2, 2)), nv_aten("f32", c(2, 2))),
    "Don't use AbtractTensors as inputs"
  )
})
