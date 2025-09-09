#' @include array.R
NULL

aval <- S7::new_generic("aval", "x", function(x) {
  S7::S7_dispatch()
})
