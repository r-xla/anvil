# length(grads) == length(outputs)
p_add[["backward"]] <- function(inputs, outputs, grads, .required) {
  list(
    if (.required[[1L]]) grads[[1L]],
    if (.required[[2L]]) grad[[2L]]
  )
}

p_mul[["backward"]] <- function(inputs, outputs, grad, .required) {
  keep(.required, grad, grad)
}
