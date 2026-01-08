# Converters between stablehlo and anvil types

# AbstractTensor -> ValueType
at2vt <- function(x) {
  stopifnot(inherits(x, "AbstractTensor"))
  stablehlo::ValueType(stablehlo::TensorType(x$dtype, x$shape))
}

# ValueType -> Abstract Tensor
vt2at <- function(x) {
  stopifnot(inherits(x, "ValueType"))
  stopifnot(inherits(x$type, "TensorType"))
  AbstractTensor(x$type$dtype, x$type$shape)
}
