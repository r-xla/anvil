# Package index

## Tensor Creation

Functions for creating and initializing tensors

- [`nv_tensor()`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
  [`nv_scalar()`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
  [`nv_empty()`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
  : AnvilTensor
- [`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md) :
  Tensor-like Objects
- [`nv_fill()`](https://r-xla.github.io/anvil/reference/nv_fill.md) :
  Constant
- [`nv_iota()`](https://r-xla.github.io/anvil/reference/nv_iota.md) :
  Iota
- [`nv_aten()`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
  [`AbstractTensor()`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
  : Abstract Tensor Class

## Tensor attributes and converters

Functions for querying tensor attributes and converting them

- [`reexports`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`shape`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`device`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`as_array`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`as_raw`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`dtype`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`ndims`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`is_dtype`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`as_dtype`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`platform`](https://r-xla.github.io/anvil/reference/reexports.md)
  [`Shape`](https://r-xla.github.io/anvil/reference/reexports.md) :
  Objects exported from other packages
- [`ambiguous()`](https://r-xla.github.io/anvil/reference/ambiguous.md)
  : Get Ambiguity of a Tensor
- [`platform(`*`<AbstractTensor>`*`)`](https://r-xla.github.io/anvil/reference/platform.AbstractTensor.md)
  : Platform for AbstractTensor
- [`platform(`*`<ConcreteTensor>`*`)`](https://r-xla.github.io/anvil/reference/platform.ConcreteTensor.md)
  : Platform for ConcreteTensor

## Tensor Serialization

Functions for serializing and deserializing tensors

- [`nv_write()`](https://r-xla.github.io/anvil/reference/nv_serialization.md)
  [`nv_read()`](https://r-xla.github.io/anvil/reference/nv_serialization.md)
  [`nv_serialize()`](https://r-xla.github.io/anvil/reference/nv_serialization.md)
  [`nv_unserialize()`](https://r-xla.github.io/anvil/reference/nv_serialization.md)
  : Tensor serialization and I/O

## Type conversion and promotion

Functions for type conversion and promotion

- [`nv_convert()`](https://r-xla.github.io/anvil/reference/nv_convert.md)
  : Convert Tensor to Different Data Type
- [`nv_bitcast_convert()`](https://r-xla.github.io/anvil/reference/nv_bitcast_convert.md)
  : Bitcast Conversion
- [`nv_promote_to_common()`](https://r-xla.github.io/anvil/reference/nv_promote_to_common.md)
  : Promote Tensors to a Common Dtype
- [`nv_broadcast_scalars()`](https://r-xla.github.io/anvil/reference/nv_broadcast_scalars.md)
  : Broadcast Scalars to Common Shape
- [`nv_broadcast_tensors()`](https://r-xla.github.io/anvil/reference/nv_broadcast_tensors.md)
  : Broadcast Tensors to a Common Shape
- [`nv_broadcast_to()`](https://r-xla.github.io/anvil/reference/nv_broadcast_to.md)
  : Broadcast
- [`common_dtype()`](https://r-xla.github.io/anvil/reference/common_dtype.md)
  : Type Promotion Rules

## Tensor manipulation

Functions for reshaping and rearranging tensors

- [`nv_reshape()`](https://r-xla.github.io/anvil/reference/nv_reshape.md)
  : Reshape
- [`nv_transpose()`](https://r-xla.github.io/anvil/reference/nv_transpose.md)
  [`t(`*`<AnvilBox>`*`)`](https://r-xla.github.io/anvil/reference/nv_transpose.md)
  : Transpose
- [`nv_concatenate()`](https://r-xla.github.io/anvil/reference/nv_concatenate.md)
  : Concatenate
- [`nv_static_slice()`](https://r-xla.github.io/anvil/reference/nv_static_slice.md)
  : Slice
- [`nv_pad()`](https://r-xla.github.io/anvil/reference/nv_pad.md) : Pad
- [`nv_reverse()`](https://r-xla.github.io/anvil/reference/nv_reverse.md)
  : Reverse

## Arithmetic operations

Basic arithmetic operations on tensors

- [`nv_add()`](https://r-xla.github.io/anvil/reference/nv_add.md) :
  Addition
- [`nv_sub()`](https://r-xla.github.io/anvil/reference/nv_sub.md) :
  Subtraction
- [`nv_mul()`](https://r-xla.github.io/anvil/reference/nv_mul.md) :
  Multiplication
- [`nv_div()`](https://r-xla.github.io/anvil/reference/nv_div.md) :
  Division
- [`nv_pow()`](https://r-xla.github.io/anvil/reference/nv_pow.md) :
  Power
- [`nv_negate()`](https://r-xla.github.io/anvil/reference/nv_negate.md)
  : Negation
- [`nv_remainder()`](https://r-xla.github.io/anvil/reference/nv_remainder.md)
  : Remainder

## Comparison operations

Element-wise comparison operations

- [`nv_eq()`](https://r-xla.github.io/anvil/reference/nv_eq.md) : Equal
- [`nv_ne()`](https://r-xla.github.io/anvil/reference/nv_ne.md) : Not
  Equal
- [`nv_gt()`](https://r-xla.github.io/anvil/reference/nv_gt.md) :
  Greater Than
- [`nv_ge()`](https://r-xla.github.io/anvil/reference/nv_ge.md) :
  Greater Than or Equal
- [`nv_lt()`](https://r-xla.github.io/anvil/reference/nv_lt.md) : Less
  Than
- [`nv_le()`](https://r-xla.github.io/anvil/reference/nv_le.md) : Less
  Than or Equal

## Mathematical functions

Mathematical and trigonometric functions

- [`nv_abs()`](https://r-xla.github.io/anvil/reference/nv_abs.md) :
  Absolute Value
- [`nv_sqrt()`](https://r-xla.github.io/anvil/reference/nv_sqrt.md) :
  Square Root
- [`nv_rsqrt()`](https://r-xla.github.io/anvil/reference/nv_rsqrt.md) :
  Reciprocal Square Root
- [`nv_cbrt()`](https://r-xla.github.io/anvil/reference/nv_cbrt.md) :
  Cube Root
- [`nv_exp()`](https://r-xla.github.io/anvil/reference/nv_exp.md) :
  Exponential
- [`nv_expm1()`](https://r-xla.github.io/anvil/reference/nv_expm1.md) :
  Exponential Minus One
- [`nv_log()`](https://r-xla.github.io/anvil/reference/nv_log.md) :
  Natural Logarithm
- [`nv_log1p()`](https://r-xla.github.io/anvil/reference/nv_log1p.md) :
  Log Plus One
- [`nv_sine()`](https://r-xla.github.io/anvil/reference/nv_sine.md) :
  Sine
- [`nv_cosine()`](https://r-xla.github.io/anvil/reference/nv_cosine.md)
  : Cosine
- [`nv_tan()`](https://r-xla.github.io/anvil/reference/nv_tan.md) :
  Tangent
- [`nv_tanh()`](https://r-xla.github.io/anvil/reference/nv_tanh.md) :
  Hyperbolic Tangent
- [`nv_atan2()`](https://r-xla.github.io/anvil/reference/nv_atan2.md) :
  Arctangent 2
- [`nv_sign()`](https://r-xla.github.io/anvil/reference/nv_sign.md) :
  Sign
- [`nv_floor()`](https://r-xla.github.io/anvil/reference/nv_floor.md) :
  Floor
- [`nv_ceil()`](https://r-xla.github.io/anvil/reference/nv_ceil.md) :
  Ceiling
- [`nv_round()`](https://r-xla.github.io/anvil/reference/nv_round.md) :
  Round
- [`nv_logistic()`](https://r-xla.github.io/anvil/reference/nv_logistic.md)
  : Logistic (Sigmoid)
- [`nv_is_finite()`](https://r-xla.github.io/anvil/reference/nv_is_finite.md)
  : Is Finite

## Reduction operations

Operations that reduce tensor dimensions

- [`nv_reduce_sum()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)
  [`nv_reduce_mean()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)
  [`nv_reduce_prod()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)
  [`nv_reduce_max()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)
  [`nv_reduce_min()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)
  [`nv_reduce_any()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)
  [`nv_reduce_all()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)
  : Reduction Operators

## Linear algebra

Linear algebra operations

- [`nv_matmul()`](https://r-xla.github.io/anvil/reference/nv_matmul.md)
  : Matrix Multiplication

## Logical and bitwise operations

Logical and bitwise operations on tensors

- [`nv_and()`](https://r-xla.github.io/anvil/reference/nv_and.md) :
  Logical And
- [`nv_or()`](https://r-xla.github.io/anvil/reference/nv_or.md) :
  Logical Or
- [`nv_xor()`](https://r-xla.github.io/anvil/reference/nv_xor.md) :
  Logical Xor
- [`nv_not()`](https://r-xla.github.io/anvil/reference/nv_not.md) :
  Logical Not
- [`nv_shift_left()`](https://r-xla.github.io/anvil/reference/nv_shift_left.md)
  : Shift Left
- [`nv_shift_right_logical()`](https://r-xla.github.io/anvil/reference/nv_shift_right_logical.md)
  : Logical Shift Right
- [`nv_shift_right_arithmetic()`](https://r-xla.github.io/anvil/reference/nv_shift_right_arithmetic.md)
  : Arithmetic Shift Right
- [`nv_popcnt()`](https://r-xla.github.io/anvil/reference/nv_popcnt.md)
  : Population Count

## Element-wise operations

Other element-wise tensor operations

- [`nv_min()`](https://r-xla.github.io/anvil/reference/nv_min.md) :
  Minimum
- [`nv_max()`](https://r-xla.github.io/anvil/reference/nv_max.md) :
  Maximum
- [`nv_clamp()`](https://r-xla.github.io/anvil/reference/nv_clamp.md) :
  Clamp

## Control flow

Control flow operations

- [`nv_if()`](https://r-xla.github.io/anvil/reference/nv_if.md) : If
- [`nv_while()`](https://r-xla.github.io/anvil/reference/nv_while.md) :
  While
- [`nv_select()`](https://r-xla.github.io/anvil/reference/nv_select.md)
  : Select

## Random number generation

Functions for generating random numbers

- [`nv_unif_rand()`](https://r-xla.github.io/anvil/reference/nv_runif.md)
  [`nv_runif()`](https://r-xla.github.io/anvil/reference/nv_runif.md) :
  Sample from a Uniform Distribution
- [`nv_rnorm()`](https://r-xla.github.io/anvil/reference/nv_rnorm.md) :
  Sample from a Normal Distribution
- [`nv_rbinom()`](https://r-xla.github.io/anvil/reference/nv_rbinom.md)
  : Sample from a Binomial Distribution
- [`nv_rdunif()`](https://r-xla.github.io/anvil/reference/nv_rdunif.md)
  : Sample from a Discrete Uniform Distribution
- [`nv_rng_state()`](https://r-xla.github.io/anvil/reference/nv_rng_state.md)
  : Generate random state

## Transformations

Code transformations

- [`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md) :
  Trace an R function into a Graph
- [`gradient()`](https://r-xla.github.io/anvil/reference/gradient.md)
  [`value_and_gradient()`](https://r-xla.github.io/anvil/reference/gradient.md)
  : Gradient
- [`transform_gradient()`](https://r-xla.github.io/anvil/reference/transform_gradient.md)
  : Transform a graph to its gradient
- [`jit()`](https://r-xla.github.io/anvil/reference/jit.md) : JIT
  compile a function
- [`jit_eval()`](https://r-xla.github.io/anvil/reference/jit_eval.md) :
  Jit an Evaluate an Expression
- [`stablehlo()`](https://r-xla.github.io/anvil/reference/stablehlo.md)
  : Lower a function to StableHLO

## Debugging

Debugging utilities and tools

- [`debug_box()`](https://r-xla.github.io/anvil/reference/debug_box.md)
  : Create a Debug Box
- [`nv_print()`](https://r-xla.github.io/anvil/reference/nv_print.md) :
  Print Tensor
- [`DebugBox()`](https://r-xla.github.io/anvil/reference/DebugBox.md) :
  Debug Box Class

## Internal Data Structures and Functions

Internal data structures and functions

- [`shape_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)
  [`ndims_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)
  [`ambiguous_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)
  [`dtype_abstract()`](https://r-xla.github.io/anvil/reference/abstract_properties.md)
  : Abstract Properties
- [`to_abstract()`](https://r-xla.github.io/anvil/reference/to_abstract.md)
  : Convert to Abstract Tensor
- [`GraphDescriptor()`](https://r-xla.github.io/anvil/reference/GraphDescriptor.md)
  : Graph Descriptor
- [`GraphValue()`](https://r-xla.github.io/anvil/reference/GraphValue.md)
  : Graph Value
- [`AnvilBox`](https://r-xla.github.io/anvil/reference/AnvilBox.md) :
  AnvilBox
- [`AnvilGraph()`](https://r-xla.github.io/anvil/reference/AnvilGraph.md)
  : Graph of Primitive Calls
- [`GraphNode`](https://r-xla.github.io/anvil/reference/GraphNode.md) :
  Graph Node
- [`GraphBox()`](https://r-xla.github.io/anvil/reference/GraphBox.md) :
  Graph Box
- [`GraphLiteral()`](https://r-xla.github.io/anvil/reference/GraphLiteral.md)
  : Graph Literal
- [`graph_desc_add()`](https://r-xla.github.io/anvil/reference/graph_desc_add.md)
  : Add a Primitive Call to a Graph Descriptor
- [`local_descriptor()`](https://r-xla.github.io/anvil/reference/local_descriptor.md)
  : Create a graph
- [`.current_descriptor()`](https://r-xla.github.io/anvil/reference/dot-current_descriptor.md)
  : Get the current graph
- [`subgraphs()`](https://r-xla.github.io/anvil/reference/subgraphs.md)
  : Get Subgraphs from Higher-Order Primitive
- [`prim()`](https://r-xla.github.io/anvil/reference/prim.md) : Get a
  Primitive
- [`AnvilPrimitive()`](https://r-xla.github.io/anvil/reference/AnvilPrimitive.md)
  : AnvilPrimitive
- [`PrimitiveCall()`](https://r-xla.github.io/anvil/reference/PrimitiveCall.md)
  : Primitive Call
- [`register_primitive()`](https://r-xla.github.io/anvil/reference/register_primitive.md)
  : Register a Primitive
- [`nv_aten()`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
  [`AbstractTensor()`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
  : Abstract Tensor Class
- [`ConcreteTensor()`](https://r-xla.github.io/anvil/reference/ConcreteTensor.md)
  : Concrete Tensor Class
- [`LiteralTensor()`](https://r-xla.github.io/anvil/reference/LiteralTensor.md)
  : Literal Tensor Class
- [`eq_type()`](https://r-xla.github.io/anvil/reference/eq_type.md)
  [`neq_type()`](https://r-xla.github.io/anvil/reference/eq_type.md) :
  Compare AbstractTensor Types

## Tree utilities

Utilities for working with nested structures

- [`flatten()`](https://r-xla.github.io/anvil/reference/flatten.md) :
  Flatten
- [`unflatten()`](https://r-xla.github.io/anvil/reference/unflatten.md)
  : Unflatten
- [`build_tree()`](https://r-xla.github.io/anvil/reference/build_tree.md)
  : Build Tree
- [`reindex_tree()`](https://r-xla.github.io/anvil/reference/reindex_tree.md)
  : Reindex Tree
- [`tree_size()`](https://r-xla.github.io/anvil/reference/tree_size.md)
  : Tree Size

## Tensor classes

Core tensor classes and constructors

## Primitives

Low-level primitive operations (nvl\_\* functions)

- [`nvl_abs()`](https://r-xla.github.io/anvil/reference/nvl_abs.md) :
  Primitive Absolute Value
- [`nvl_add()`](https://r-xla.github.io/anvil/reference/nvl_add.md) :
  Primitive Addition
- [`nvl_and()`](https://r-xla.github.io/anvil/reference/nvl_and.md) :
  Primitive And
- [`nvl_atan2()`](https://r-xla.github.io/anvil/reference/nvl_atan2.md)
  : Primitive Atan2
- [`nvl_bitcast_convert()`](https://r-xla.github.io/anvil/reference/nvl_bitcast_convert.md)
  : Primitive Bitcast Convert
- [`nvl_broadcast_in_dim()`](https://r-xla.github.io/anvil/reference/nvl_broadcast_in_dim.md)
  : Primitive Broadcast
- [`nvl_cbrt()`](https://r-xla.github.io/anvil/reference/nvl_cbrt.md) :
  Primitive Cube Root
- [`nvl_ceil()`](https://r-xla.github.io/anvil/reference/nvl_ceil.md) :
  Primitive Ceiling
- [`nvl_clamp()`](https://r-xla.github.io/anvil/reference/nvl_clamp.md)
  : Primitive Clamp
- [`nvl_concatenate()`](https://r-xla.github.io/anvil/reference/nvl_concatenate.md)
  : Primitive Concatenate
- [`nvl_convert()`](https://r-xla.github.io/anvil/reference/nvl_convert.md)
  : Primitive Convert
- [`nvl_cosine()`](https://r-xla.github.io/anvil/reference/nvl_cosine.md)
  : Primitive Cosine
- [`nvl_div()`](https://r-xla.github.io/anvil/reference/nvl_div.md) :
  Primitive Division
- [`nvl_dot_general()`](https://r-xla.github.io/anvil/reference/nvl_dot_general.md)
  : Primitive Dot General
- [`nvl_dynamic_slice()`](https://r-xla.github.io/anvil/reference/nvl_dynamic_slice.md)
  : Primitive Dynamic Slice
- [`nvl_dynamic_update_slice()`](https://r-xla.github.io/anvil/reference/nvl_dynamic_update_slice.md)
  : Primitive Dynamic Update Slice
- [`nvl_eq()`](https://r-xla.github.io/anvil/reference/nvl_eq.md) :
  Primitive Equal
- [`nvl_exp()`](https://r-xla.github.io/anvil/reference/nvl_exp.md) :
  Primitive Exponential
- [`nvl_expm1()`](https://r-xla.github.io/anvil/reference/nvl_expm1.md)
  : Primitive Exponential Minus One
- [`nvl_fill()`](https://r-xla.github.io/anvil/reference/nvl_fill.md) :
  Primitive Fill
- [`nvl_floor()`](https://r-xla.github.io/anvil/reference/nvl_floor.md)
  : Primitive Floor
- [`nvl_ge()`](https://r-xla.github.io/anvil/reference/nvl_ge.md) :
  Primitive Greater Equal
- [`nvl_gt()`](https://r-xla.github.io/anvil/reference/nvl_gt.md) :
  Primitive Greater Than
- [`nvl_if()`](https://r-xla.github.io/anvil/reference/nvl_if.md) :
  Primitive If
- [`nvl_iota()`](https://r-xla.github.io/anvil/reference/nvl_iota.md) :
  Primitive Iota
- [`nvl_is_finite()`](https://r-xla.github.io/anvil/reference/nvl_is_finite.md)
  : Primitive Is Finite
- [`nvl_le()`](https://r-xla.github.io/anvil/reference/nvl_le.md) :
  Primitive Less Equal
- [`nvl_log()`](https://r-xla.github.io/anvil/reference/nvl_log.md) :
  Primitive Logarithm
- [`nvl_log1p()`](https://r-xla.github.io/anvil/reference/nvl_log1p.md)
  : Primitive Log Plus One
- [`nvl_logistic()`](https://r-xla.github.io/anvil/reference/nvl_logistic.md)
  : Primitive Logistic (Sigmoid)
- [`nvl_lt()`](https://r-xla.github.io/anvil/reference/nvl_lt.md) :
  Primitive Less Than
- [`nvl_max()`](https://r-xla.github.io/anvil/reference/nvl_max.md) :
  Primitive Maximum
- [`nvl_min()`](https://r-xla.github.io/anvil/reference/nvl_min.md) :
  Primitive Minimum
- [`nvl_mul()`](https://r-xla.github.io/anvil/reference/nvl_mul.md) :
  Primitive Multiplication
- [`nvl_ne()`](https://r-xla.github.io/anvil/reference/nvl_ne.md) :
  Primitive Not Equal
- [`nvl_negate()`](https://r-xla.github.io/anvil/reference/nvl_negate.md)
  : Primitive Negation
- [`nvl_not()`](https://r-xla.github.io/anvil/reference/nvl_not.md) :
  Primitive Not
- [`nvl_or()`](https://r-xla.github.io/anvil/reference/nvl_or.md) :
  Primitive Or
- [`nvl_pad()`](https://r-xla.github.io/anvil/reference/nvl_pad.md) :
  Primitive Pad
- [`nvl_popcnt()`](https://r-xla.github.io/anvil/reference/nvl_popcnt.md)
  : Primitive Population Count
- [`nvl_pow()`](https://r-xla.github.io/anvil/reference/nvl_pow.md) :
  Primitive Power
- [`nvl_print()`](https://r-xla.github.io/anvil/reference/nvl_print.md)
  : Primitive Print
- [`nvl_reduce_all()`](https://r-xla.github.io/anvil/reference/nvl_reduce_all.md)
  : Primitive All Reduction
- [`nvl_reduce_any()`](https://r-xla.github.io/anvil/reference/nvl_reduce_any.md)
  : Primitive Any Reduction
- [`nvl_reduce_max()`](https://r-xla.github.io/anvil/reference/nvl_reduce_max.md)
  : Primitive Max Reduction
- [`nvl_reduce_min()`](https://r-xla.github.io/anvil/reference/nvl_reduce_min.md)
  : Primitive Min Reduction
- [`nvl_reduce_prod()`](https://r-xla.github.io/anvil/reference/nvl_reduce_prod.md)
  : Primitive Product Reduction
- [`nvl_reduce_sum()`](https://r-xla.github.io/anvil/reference/nvl_reduce_sum.md)
  : Primitive Sum Reduction
- [`nvl_remainder()`](https://r-xla.github.io/anvil/reference/nvl_remainder.md)
  : Primitive Remainder
- [`nvl_reshape()`](https://r-xla.github.io/anvil/reference/nvl_reshape.md)
  : Primitive Reshape
- [`nvl_reverse()`](https://r-xla.github.io/anvil/reference/nvl_reverse.md)
  : Primitive Reverse
- [`nvl_rng_bit_generator()`](https://r-xla.github.io/anvil/reference/nvl_rng_bit_generator.md)
  : Primitive RNG Bit Generator
- [`nvl_round()`](https://r-xla.github.io/anvil/reference/nvl_round.md)
  : Primitive Round
- [`nvl_rsqrt()`](https://r-xla.github.io/anvil/reference/nvl_rsqrt.md)
  : Primitive Reciprocal Square Root
- [`nvl_select()`](https://r-xla.github.io/anvil/reference/nvl_select.md)
  : Primitive Select
- [`nvl_shift_left()`](https://r-xla.github.io/anvil/reference/nvl_shift_left.md)
  : Primitive Shift Left
- [`nvl_shift_right_arithmetic()`](https://r-xla.github.io/anvil/reference/nvl_shift_right_arithmetic.md)
  : Primitive Arithmetic Shift Right
- [`nvl_shift_right_logical()`](https://r-xla.github.io/anvil/reference/nvl_shift_right_logical.md)
  : Primitive Logical Shift Right
- [`nvl_sign()`](https://r-xla.github.io/anvil/reference/nvl_sign.md) :
  Primitive Sign
- [`nvl_sine()`](https://r-xla.github.io/anvil/reference/nvl_sine.md) :
  Primitive Sine
- [`nvl_sqrt()`](https://r-xla.github.io/anvil/reference/nvl_sqrt.md) :
  Primitive Square Root
- [`nvl_static_slice()`](https://r-xla.github.io/anvil/reference/nvl_static_slice.md)
  : Primitive Static Slice
- [`nvl_sub()`](https://r-xla.github.io/anvil/reference/nvl_sub.md) :
  Primitive Subtraction
- [`nvl_tan()`](https://r-xla.github.io/anvil/reference/nvl_tan.md) :
  Primitive Tangent
- [`nvl_tanh()`](https://r-xla.github.io/anvil/reference/nvl_tanh.md) :
  Primitive Hyperbolic Tangent
- [`nvl_transpose()`](https://r-xla.github.io/anvil/reference/nvl_transpose.md)
  : Primitive Transpose
- [`nvl_while()`](https://r-xla.github.io/anvil/reference/nvl_while.md)
  : Primitive While Loop
- [`nvl_xor()`](https://r-xla.github.io/anvil/reference/nvl_xor.md) :
  Primitive Xor
