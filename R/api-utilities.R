#' @title Generate random state
#' @name nv_rng_state
#' @description
#' lightweight function to generate an initial state
#' @param seed (`integer(1)`)\cr
#'   Seed value
#' @return [`nv_tensor`] of dtype `ui64` and shape (2)
#' @export
nv_rng_state <- function(seed) {
  checkmate::assert_int(seed)
  .nv_rng_state(nv_scalar(seed, dtype = "i32"))
}

#' @include jit.R
.nv_rng_state <- jit(function(state) {
  state <- nv_bitcast_convert(state, dtype = "ui16")
  nv_convert(state, "ui64")
})
