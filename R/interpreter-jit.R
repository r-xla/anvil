#' @export
jit <- function(f, ...) {
  args <- list(...)
  # TODO: also support params that are not ShapedTensors
  is_shaped_tensor <- sapply(args, \(arg) inherits(arg, ShapedTensor))
  shaped_tensors <- args[is_shaped_tensor]
  params <- args[!is_shaped_tensor]
}
