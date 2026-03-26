#' @title Generate RNG State
#' @name nv_rng_state
#' @description
#' Creates an initial RNG state from a seed. This state is required by all
#' random sampling functions and is updated after each call.
#' Must be called inside a [`jit()`] or [`jit_eval()`] context.
#' @param seed (`integer(1)`)\cr
#'   Seed value.
#' @return [`nv_array`] of dtype `ui64` and shape `(2)`.
#' @family rng
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   state <- nv_rng_state(42L)
#'   state
#' })
#' @export
nv_rng_state <- function(seed) {
  checkmate::assert_int(seed)
  seed <- nv_scalar(seed, dtype = "i32")
  state <- nv_bitcast_convert(seed, dtype = "ui16")
  nv_convert(state, "ui64")
}
