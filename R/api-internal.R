#' @title Internal: Random Unit Uniform Numbers
#' @name nv_runif
#' @description
#' generate random uniform numbers in [0, 1)
#' @param initial_state state seed
#' @param dtype output dtype either "f32" or "f64"
#' @param shape_out output shape
nv_unif_rand <- function(
  initial_state,
  dtype = "f64",
  shape_out
) {
  checkmate::assertChoice(dtype, c("f32", "f64"))
  checkmate::assertIntegerish(shape_out, lower = 1, min.len = 1, any.missing = FALSE)

  # generate random bits
  # use THREE_FRY as rng algorithm: JAX default
  rbits <- nv_rng_bit_generator(
    initial_state = initial_state,
    "THREE_FRY",
    paste0("ui", sub("f(\\d+)", "\\1", dtype)),
    shape_out = shape_out
  )

  # shift value: 9 for f32, 11 for f64
  shift <- nv_scalar(
    ifelse(dtype == "f32", 9L, 11L),
    dtype = paste0("ui", sub("f(\\d+)", "\\1", dtype))
  )

  # shift to the right, s.t. exponent bits are all 0
  mantissa <- nv_shift_right_logical(rbits[[2]], shift)

  # interpretation of 1.0 (float) as unsigned
  one_bits <- nv_bitcast_convert(
    nv_scalar(1.0, dtype = dtype),
    dtype = paste0("ui", sub("f(\\d+)", "\\1", dtype))
  )

  # bitwise or -> exponent from 1.0 (float), mantissa is random
  U <- nv_or(mantissa, one_bits)

  # convert back to requested dtype
  # resulting RVs  are in [1, 2)
  U <- nv_bitcast_convert(U, dtype = dtype)

  # shift to [0, 1)
  U <- nv_add(U, nv_scalar(-1, dtype = dtype))

  # return state and RVs
  list(rbits[[1]], U)
}
