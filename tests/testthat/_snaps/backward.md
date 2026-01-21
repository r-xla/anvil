# wrt for non-tensor input: gradient

    Code
      g <- gradient(nv_round, wrt = "method")
      g(nv_scalar(1), method = "nearest_even")
    Condition
      Error in `check_wrt_tensorish()`:
      x Cannot compute gradient with respect to non-tensor argument.
      i Got <character> instead of a tensor.

# wrt for non-tensor input: value_and_gradient

    Code
      g <- value_and_gradient(nv_round, wrt = "method")
      g(nv_scalar(1), method = "nearest_even")
    Condition
      Error in `check_wrt_tensorish()`:
      x Cannot compute gradient with respect to non-tensor argument.
      i Got <character> instead of a tensor.

# wrt for nested non-tensor input: gradient

    Code
      g <- gradient(f, wrt = "x")
      g(x = list(nv_scalar(1), 2L))
    Condition
      Error in `check_wrt_tensorish()`:
      x Can only compute gradient with respect to float tensors.
      i Got i32 instead of a float tensor.

# wrt for nested non-tensor input: value_and_gradient

    Code
      g <- value_and_gradient(f, wrt = "x")
      g(x = list(nv_scalar(1), 2L))
    Condition
      Error in `check_wrt_tensorish()`:
      x Can only compute gradient with respect to float tensors.
      i Got i32 instead of a float tensor.

# can only compute gradient w.r.t. float tensors

    Code
      gradient(nv_floor, wrt = "operand")(nv_scalar(1L))
    Condition
      Error in `nvl_floor()`:
      ! `operand` must have dtype FloatType.
      x Got <i32>.

