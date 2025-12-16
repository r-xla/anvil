# Converters between stablehlo and anvil types

# AbstractTensor -> FuncValue
st2fv <- function(x, func) {
  value_type <- st2vt(x)
  value_id <- stablehlo::ValueId()
  func@inputs <- stablehlo::FuncInputs(c(
    func@inputs@items,
    list(stablehlo::FuncInput(
      id = value_id,
      type = value_type
    ))
  ))
  stablehlo::FuncValue(
    value_id = value_id,
    value_type = value_type,
    func = func
  )
}

# AbstractTensor -> ValueType
st2vt <- function(x) {
  stopifnot(inherits(x, AbstractTensor))
  stablehlo::ValueType(stablehlo::TensorType(x@dtype, x@shape))
}

# ValueType -> Abstract Tensor
vt2st <- function(x) {
  stopifnot(inherits(x, stablehlo::ValueType))
  stopifnot(inherits(x@type, stablehlo::TensorType))
  AbstractTensor(x@type@dtype, x@type@shape)
}
