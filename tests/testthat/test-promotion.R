test_that("common_type_info: single argument", {
  s1 <- ShapedTensor(dtype("i32"), Shape(c(1, 2)), FALSE)
  result <- common_type_info(s1)
  expect_equal(result[[1L]], dtype("i32"))
  expect_equal(result[[2L]], FALSE)

  s2 <- ShapedTensor(dtype("f32"), Shape(c(2, 3)), TRUE)
  result <- common_type_info(s2)
  expect_equal(result[[1L]], dtype("f32"))
  expect_equal(result[[2L]], TRUE)
})

test_that("common_type_info: two arguments", {
  check <- function(dt1, dt2, a1, a2, expected_dt, expected_ambiguous) {
    s1 <- ShapedTensor(dt1, Shape(c(1, 2)), a1)
    s2 <- ShapedTensor(dt2, Shape(c(2, 1)), a2)
    result <- common_type_info(s1, s2)
    expect_equal(result[[1L]], expected_dt)
    expect_equal(result[[2L]], expected_ambiguous)
    # Check symmetry for dtype (ambiguity may differ based on order in some edge cases)
    result_rev <- common_type_info(s2, s1)
    expect_equal(result_rev[[1L]], expected_dt)
  }

  # both are ambiguous -> result is ambiguous

  check(dtype("i32"), dtype("i32"), TRUE, TRUE, dtype("i32"), TRUE)
  check(dtype("i32"), dtype("f32"), TRUE, TRUE, dtype("f32"), TRUE)

  # one is ambiguous
  # ambiguous float + known int -> ambiguous float (ambiguous wins because it's float)
  check(dtype("f32"), dtype("i32"), TRUE, FALSE, dtype("f32"), TRUE)
  # ambiguous int + known float -> known float (known wins)
  check(dtype("i32"), dtype("f32"), TRUE, FALSE, dtype("f32"), FALSE)
  # both types same -> known wins

  check(dtype("i32"), dtype("i32"), TRUE, FALSE, dtype("i32"), FALSE)

  # neither is ambiguous -> result is not ambiguous
  check(dtype("f32"), dtype("i32"), FALSE, FALSE, dtype("f32"), FALSE)
  check(dtype("f32"), dtype("f64"), FALSE, FALSE, dtype("f64"), FALSE)
  check(dtype("ui32"), dtype("i32"), FALSE, FALSE, dtype("i64"), FALSE)
})

test_that("common_type_info: multiple arguments", {
  i32 <- ShapedTensor(dtype("i32"), Shape(1), FALSE)
  f32 <- ShapedTensor(dtype("f32"), Shape(2), FALSE)
  f64 <- ShapedTensor(dtype("f64"), Shape(3), FALSE)

  result <- common_type_info(i32, f32, f64)
  expect_equal(result[[1L]], dtype("f64"))
  expect_equal(result[[2L]], FALSE)

  result <- common_type_info(f64, f32, i32)
  expect_equal(result[[1L]], dtype("f64"))
  expect_equal(result[[2L]], FALSE)

  result <- common_type_info(i32, i32, i32)
  expect_equal(result[[1L]], dtype("i32"))
  expect_equal(result[[2L]], FALSE)

  # With ambiguous types
  i32_amb <- ShapedTensor(dtype("i32"), Shape(1), TRUE)
  i64_known <- ShapedTensor(dtype("i64"), Shape(2), FALSE)

  result <- common_type_info(i32_amb, i64_known)
  expect_equal(result[[1L]], dtype("i64"))
  expect_equal(result[[2L]], FALSE)
})

test_that("common_type_info: error on no arguments", {
  expect_error(common_type_info(), "No arguments provided")
})

test_that("promote_dt_known", {
  check <- function(dt1, dt2, dt3) {
    expect_equal(
      promote_dt_known(dt1, dt2),
      dt3
    )
    expect_equal(
      promote_dt_known(dt2, dt1),
      dt3
    )
  }

  check(dtype("f64"), dtype("f64"), dtype("f64"))
  check(dtype("i32"), dtype("i32"), dtype("i32"))
  check(dtype("i1"), dtype("i1"), dtype("i1"))

  # floats dominate
  check(dtype("f64"), dtype("f32"), dtype("f64"))
  check(dtype("f64"), dtype("i32"), dtype("f64"))
  check(dtype("f32"), dtype("i32"), dtype("f32"))
  check(dtype("f32"), dtype("i1"), dtype("f32"))

  # signed ints
  check(dtype("i32"), dtype("i16"), dtype("i32"))
  check(dtype("i64"), dtype("i32"), dtype("i64"))
  check(dtype("i64"), dtype("i16"), dtype("i64"))
  check(dtype("i64"), dtype("i1"), dtype("i64"))
  # against unsigned ints
  check(dtype("i32"), dtype("ui8"), dtype("i32"))
  check(dtype("i32"), dtype("ui32"), dtype("i64"))
  check(dtype("i64"), dtype("ui64"), dtype("i64"))
  # unsigned vs unsigned
  check(dtype("ui64"), dtype("ui32"), dtype("ui64"))
})

test_that("promote_dt_ambiguous", {
  check <- function(x, y, z) {
    expect_equal(
      promote_dt_ambiguous(x, y),
      z
    )
    expect_equal(
      promote_dt_ambiguous(y, x),
      z
    )
  }
  check(dtype("i32"), dtype("i32"), dtype("i32"))
  check(dtype("f32"), dtype("f32"), dtype("f32"))
  check(dtype("i1"), dtype("i1"), dtype("i1"))

  check(dtype("i32"), dtype("f32"), dtype("f32"))
  check(dtype("i1"), dtype("f32"), dtype("f32"))
  check(dtype("i1"), dtype("i32"), dtype("i32"))
})

test_that("promote_dt_ambiguous_to_known", {
  check <- function(amb, known, z) {
    expect_equal(
      promote_dt_ambiguous_to_known(amb, known),
      z
    )
  }
  check(dtype("i32"), dtype("i32"), dtype("i32"))
  check(dtype("i1"), dtype("i1"), dtype("i1"))
  check(dtype("f32"), dtype("f32"), dtype("f32"))
  # ambiguous can only be i32, f32 or bool

  check(dtype("i32"), dtype("i8"), dtype("i8"))
  check(dtype("i32"), dtype("i64"), dtype("i64"))
  check(dtype("i32"), dtype("i1"), dtype("i32"))
  check(dtype("i64"), dtype("f32"), dtype("f32"))

  check(dtype("f32"), dtype("f64"), dtype("f64"))
  check(dtype("f64"), dtype("f32"), dtype("f32"))
  check(dtype("f32"), dtype("i32"), dtype("f32"))

  check(dtype("i1"), dtype("f32"), dtype("f32"))
  check(dtype("i1"), dtype("i32"), dtype("i32"))
})
