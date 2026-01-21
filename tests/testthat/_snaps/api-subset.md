# nv_subset_assign stablehlo: promotes correctly

    Code
      jit_eval({
        x <- nv_tensor(1:3)
        x[1] <- nv_tensor(1)
        x
      })
    Condition
      Error in `nv_subset_assign()`:
      ! Value type f32 is not promotable to left-hand side type i32

