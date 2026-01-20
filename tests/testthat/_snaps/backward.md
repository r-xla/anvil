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
      x Cannot compute gradient with respect to non-tensor argument.
      i Got <integer> instead of a tensor.

# wrt for nested non-tensor input: value_and_gradient

    Code
      g <- value_and_gradient(f, wrt = "x")
      g(x = list(nv_scalar(1), 2L))
    Condition
      Error in `check_wrt_tensorish()`:
      x Cannot compute gradient with respect to non-tensor argument.
      i Got <integer> instead of a tensor.

