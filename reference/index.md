# Package index

## All functions

- [`AbstractTensor()`](AbstractTensor.md) : Abstract Tensor Class

- [`ConcreteTensor()`](ConcreteTensor.md) : Concrete Tensor Class

- [`Graph()`](Graph.md) : Graph of Primitive Calls

- [`GraphDescriptor()`](GraphDescriptor.md) : Graph Descriptor

- [`GraphLiteral()`](GraphLiteral.md) : Graph Literal

- [`GraphValue()`](GraphValue.md) : Graph Value

- [`LiteralTensor()`](LiteralTensor.md) : Literal Tensor Class

- [`Primitive()`](Primitive.md) : Primitive

- [`PrimitiveCall()`](PrimitiveCall.md) : Primitive Call

- [`common_dtype()`](common_dtype.md) : Type Promotion Rules

- [`.current_descriptor()`](dot-current_descriptor.md) : Get the current
  graph

- [`gradient()`](gradient.md) [`value_and_gradient()`](gradient.md) :
  Gradient

- [`jit()`](jit.md) : JIT compile a function

- [`local_descriptor()`](local_descriptor.md) : Create a graph

- [`mut()`](mut.md) :

  Convert an `S7` class to a mutable `S7` object

- [`nv_add()`](nv_binary_ops.md) [`nv_mul()`](nv_binary_ops.md)
  [`nv_sub()`](nv_binary_ops.md) [`nv_div()`](nv_binary_ops.md)
  [`nv_pow()`](nv_binary_ops.md) [`nv_eq()`](nv_binary_ops.md)
  [`nv_ne()`](nv_binary_ops.md) [`nv_gt()`](nv_binary_ops.md)
  [`nv_ge()`](nv_binary_ops.md) [`nv_lt()`](nv_binary_ops.md)
  [`nv_le()`](nv_binary_ops.md) [`nv_max()`](nv_binary_ops.md)
  [`nv_min()`](nv_binary_ops.md) [`nv_remainder()`](nv_binary_ops.md)
  [`nv_and()`](nv_binary_ops.md) [`nv_or()`](nv_binary_ops.md)
  [`nv_xor()`](nv_binary_ops.md) [`nv_shift_left()`](nv_binary_ops.md)
  [`nv_shift_right_logical()`](nv_binary_ops.md)
  [`nv_shift_right_arithmetic()`](nv_binary_ops.md)
  [`nv_atan2()`](nv_binary_ops.md) : Binary Operations

- [`nv_bitcast_convert()`](nv_bitcast_convert.md) : Bitcast Conversion

- [`nv_broadcast_scalars()`](nv_broadcast_scalars.md) : Broadcast
  Scalars to Common Shape

- [`nv_broadcast_tensors()`](nv_broadcast_tensors.md) : Broadcast
  Tensors to a Common Shape

- [`nv_broadcast_to()`](nv_broadcast_to.md) : Broadcast

- [`nv_concatenate()`](nv_concatenate.md) : Concatenate

- [`nv_convert()`](nv_convert.md) : Convert Tensor to Different Data
  Type

- [`nv_fill()`](nv_fill.md) : Constant

- [`nv_if()`](nv_if.md) : If

- [`nv_matmul()`](nv_matmul.md) : Matrix Multiplication

- [`nv_promote_to_common()`](nv_promote_to_common.md) : Promote Tensors
  to a Common Dtype

- [`nv_reduce_sum()`](nv_reduce_ops.md)
  [`nv_reduce_mean()`](nv_reduce_ops.md)
  [`nv_reduce_prod()`](nv_reduce_ops.md)
  [`nv_reduce_max()`](nv_reduce_ops.md)
  [`nv_reduce_min()`](nv_reduce_ops.md)
  [`nv_reduce_any()`](nv_reduce_ops.md)
  [`nv_reduce_all()`](nv_reduce_ops.md) : Reduction Operators

- [`nv_reshape()`](nv_reshape.md) : Reshape

- [`nv_rng_bit_generator()`](nv_rng_bit_generator.md) : Random Numbers

- [`nv_rng_state()`](nv_rng_state.md) : Generate random state

- [`nv_rnorm()`](nv_rnorm.md) : Random Normal Numbers

- [`nv_runif()`](nv_runif.md) [`nv_unif_rand()`](nv_runif.md) : Random
  Uniform Numbers

- [`nv_select()`](nv_select.md) : Select

- [`nv_slice()`](nv_slice.md) : Slice

- [`AnvilTensor`](nv_tensor.md) [`nv_tensor()`](nv_tensor.md)
  [`nv_scalar()`](nv_tensor.md) [`nv_empty()`](nv_tensor.md) : Tensor

- [`nv_transpose()`](nv_transpose.md)
  [`t(`*`<anvil::GraphBox>`*`)`](nv_transpose.md) : Transpose

- [`nv_neg()`](nv_unary_ops.md) [`nv_abs()`](nv_unary_ops.md)
  [`nv_sqrt()`](nv_unary_ops.md) [`nv_rsqrt()`](nv_unary_ops.md)
  [`nv_log()`](nv_unary_ops.md) [`nv_tanh()`](nv_unary_ops.md)
  [`nv_tan()`](nv_unary_ops.md) [`nv_sine()`](nv_unary_ops.md)
  [`nv_cosine()`](nv_unary_ops.md) [`nv_floor()`](nv_unary_ops.md)
  [`nv_ceil()`](nv_unary_ops.md) [`nv_sign()`](nv_unary_ops.md)
  [`nv_exp()`](nv_unary_ops.md) [`nv_round()`](nv_unary_ops.md) : Unary
  Operations

- [`nv_while()`](nv_while.md) : While

- [`platform()`](platform.md) : Platform

- [`stablehlo()`](stablehlo.md) : Lower a function to StableHLO

- [`to_abstract()`](to_abstract.md) : Convert to Abstract Tensor

- [`trace_fn()`](trace_fn.md) : Trace an R function into a Graph
