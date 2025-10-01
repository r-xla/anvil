# Converters between stablehlo and anvil types

# ShapedTensor -> FunctionInput (not really a type)
st2fi <- function(x, func) {
  value_type <- st2vt(x)
  value_id <- stablehlo:::ValueId()
  func@inputs <- stablehlo:::FuncInputs(c(
    func@inputs@items,
    list(stablehlo:::FuncInput(
      id = value_id,
      type = value_type
    ))
  ))
  stablehlo:::FuncVariable(
    value_id = value_id,
    value_type = value_type,
    func = func
  )
}

# ShapedTensor -> ValueType
st2vt <- function(x) {
  stopifnot(inherits(x, ShapedTensor))
  stablehlo::ValueType(stablehlo::TensorType(x@dtype, x@shape))
}

# ValueType -> Shaped Tensor
vt2st <- function(x) {
  stopifnot(inherits(x, stablehlo::ValueType))
  stopifnot(inherits(x@type, stablehlo::TensorType))
  ShapedTensor(x@type@dtype, x@type@shape)
}
