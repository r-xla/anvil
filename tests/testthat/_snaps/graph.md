# GraphLiteral

    Code
      gl
    Output
      GraphLiteral(1, i32?) 

# error handling: adds the primitive call

    Code
      jit(nvl_ceil)(nv_tensor(1:4))
    Condition
      Error in `nvl_ceil()`:
      ! `operand` must have dtype FloatType.
      x Got <i32>.

# error handling: increments index in error message

    Code
      jit(nvl_transpose, static = "permutation")(nv_tensor(1:4, shape = c(2, 2)),
      permutation = c(2, 2))
    Condition
      Error in `nvl_transpose()`:
      ! `permutation` must be a permutation of c(1, 2).
      i Got c(2, 2).

