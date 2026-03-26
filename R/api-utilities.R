#' @title Generate RNG State
#' @name nv_rng_state
#' @description
#' Creates an initial RNG state from a seed. This state is required by all
#' random sampling functions and is updated after each call.
#' @param seed ([`arrayish`])\cr
#'   Scalar `i32` seed value.
#' @return [`nv_array`] of dtype `ui64` and shape `(2)`.
#' @family rng
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   state <- nv_rng_state(nv_scalar(42L))
#'   state
#' })
#' @export
nv_rng_state <- function(seed) {
  state <- nv_bitcast_convert(seed, dtype = "ui16")
  nv_convert(state, "ui64")
}
