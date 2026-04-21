# jit: bare vector without dim errors

    Code
      f(c(1, 2, 3))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `x` to an <AnvilArray>.
      i Expected an <AnvilArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <numeric> of length 3.

# jit: non-array/non-scalar leaves (e.g. character) error

    Code
      f("hello")
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `x` to an <AnvilArray>.
      i Expected an <AnvilArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <character> of length 1.

# xla: bare vector errors

    Code
      f_compiled(c(1, 2, 3))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `x` to an <AnvilArray>.
      i Expected an <AnvilArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <numeric> of length 3.

# quickr: bare vector errors

    Code
      f(c(1, 2, 3))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `x` to an <AnvilArray>.
      i Expected an <AnvilArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <numeric> of length 3.

# jit: error shows path for nested list element

    Code
      f(list(list(a = "abc")))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `l[[1]]$a` to an <AnvilArray>.
      i Expected an <AnvilArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <character> of length 1.

# jit: error shows path for unnamed nested element

    Code
      f(list("bad", nv_scalar(1)))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `pair[[1]]` to an <AnvilArray>.
      i Expected an <AnvilArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <character> of length 1.

# xla: error shows path for nested list element

    Code
      f_compiled(list("bad", nv_scalar(1)))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `pair[[1]]` to an <AnvilArray>.
      i Expected an <AnvilArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <character> of length 1.

