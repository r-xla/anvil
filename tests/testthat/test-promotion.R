test_that("common_type_info: single argument", {
  s1 <- ShapedTensor(dt_i32, Shape(c(1, 2)), FALSE)
  result <- common_type_info(s1)
  expect_equal(result[[1L]], dt_i32)
  expect_equal(result[[2L]], FALSE)

  s2 <- ShapedTensor(dt_f32, Shape(c(2, 3)), TRUE)
  result <- common_type_info(s2)
  expect_equal(result[[1L]], dt_f32)
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

  check(dt_i32, dt_i32, TRUE, TRUE, dt_i32, TRUE)
  check(dt_i32, dt_f32, TRUE, TRUE, dt_f32, TRUE)

  # one is ambiguous
  # ambiguous float + known int -> ambiguous float (ambiguous wins because it's float)
  check(dt_f32, dt_i32, TRUE, FALSE, dt_f32, TRUE)
  # ambiguous int + known float -> known float (known wins)
  check(dt_i32, dt_f32, TRUE, FALSE, dt_f32, FALSE)
  # both types same -> known wins

  check(dt_i32, dt_i32, TRUE, FALSE, dt_i32, FALSE)

  # neither is ambiguous -> result is not ambiguous
  check(dt_f32, dt_i32, FALSE, FALSE, dt_f32, FALSE)
  check(dt_f32, dt_f64, FALSE, FALSE, dt_f64, FALSE)
  check(dt_ui32, dt_i32, FALSE, FALSE, dt_i64, FALSE)
})

test_that("common_type_info: multiple arguments", {
  i32 <- ShapedTensor(dt_i32, Shape(1), FALSE)
  f32 <- ShapedTensor(dt_f32, Shape(2), FALSE)
  f64 <- ShapedTensor(dt_f64, Shape(3), FALSE)

  result <- common_type_info(i32, f32, f64)
  expect_equal(result[[1L]], dt_f64)
  expect_equal(result[[2L]], FALSE)

  result <- common_type_info(f64, f32, i32)
  expect_equal(result[[1L]], dt_f64)
  expect_equal(result[[2L]], FALSE)

  result <- common_type_info(i32, i32, i32)
  expect_equal(result[[1L]], dt_i32)
  expect_equal(result[[2L]], FALSE)

  # With ambiguous types
  i32_amb <- ShapedTensor(dt_i32, Shape(1), TRUE)
  i64_known <- ShapedTensor(dt_i64, Shape(2), FALSE)

  result <- common_type_info(i32_amb, i64_known)
  expect_equal(result[[1L]], dt_i64)
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

  check(dt_f64, dt_f64, dt_f64)
  check(dt_i32, dt_i32, dt_i32)
  check(dt_i1, dt_i1, dt_i1)

  # floats dominate
  check(dt_f64, dt_f32, dt_f64)
  check(dt_f64, dt_i32, dt_f64)
  check(dt_f32, dt_i32, dt_f32)
  check(dt_f32, dt_i1, dt_f32)

  # signed ints
  check(dt_i32, dt_i16, dt_i32)
  check(dt_i64, dt_i32, dt_i64)
  check(dt_i64, dt_i16, dt_i64)
  check(dt_i64, dt_i1, dt_i64)
  # against unsigned ints
  check(dt_i32, dt_ui8, dt_i32)
  check(dt_i32, dt_ui32, dt_i64)
  check(dt_i64, dt_ui64, dt_i64)
  # unsigned vs unsigned
  check(dt_ui64, dt_ui32, dt_ui64)
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
  check(dt_i32, dt_i32, dt_i32)
  check(dt_f32, dt_f32, dt_f32)
  check(dt_i1, dt_i1, dt_i1)

  check(dt_i32, dt_f32, dt_f32)
  check(dt_i1, dt_f32, dt_f32)
  check(dt_i1, dt_i32, dt_i32)
})

test_that("promote_dt_ambiguous_to_known", {
  check <- function(amb, known, z) {
    expect_equal(
      promote_dt_ambiguous_to_known(amb, known),
      z
    )
  }
  check(dt_i32, dt_i32, dt_i32)
  check(dt_i1, dt_i1, dt_i1)
  check(dt_f32, dt_f32, dt_f32)
  # ambiguous can only be i32, f32 or bool

  check(dt_i32, dt_i8, dt_i8)
  check(dt_i32, dt_i64, dt_i64)
  check(dt_i32, dt_i1, dt_i32)
  check(dt_i64, dt_f32, dt_f32)

  check(dt_f32, dt_f64, dt_f64)
  check(dt_f64, dt_f32, dt_f32)
  check(dt_f32, dt_i32, dt_f32)

  check(dt_i1, dt_f32, dt_f32)
  check(dt_i1, dt_i32, dt_i32)
})
