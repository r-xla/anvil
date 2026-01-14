# GraphLiteral

    Code
      gl
    Output
      GraphLiteral(1, i32?) 

# error handling: adds the primitive call

    Code
      nvl_add(nv_tensor(1), nv_tensor(1:4))
    Condition
      Error in `nvl_add()`:
      ! `lhs` and `rhs` must have the same tensor type.
      i Inputs:
      lhs = f32[1]
      rhs = i32[4]

# error handling: increments index in error message

    Code
      nvl_transpose(nv_tensor(1:4, shape = c(2, 2)), permutation = c(2, 2))
    Condition
      Error in `nvl_transpose()`:
      ! `permutation` must be a permutation of c(1, 2).
      i Got c(2, 2).
      i Inputs:
        operand = i32[2, 2]
      i Params:
        [permutation = c(2, 2)]

