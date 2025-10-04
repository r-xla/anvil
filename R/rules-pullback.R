# Each pullback rule needs to define how to do the forward pass
# and backward pass
# Important: Don't use infix operators here, as they correspond to
# the broadcasted version

mul_bwd <- function(grad, operand) {
  # grad is of shape [du1, ..., dun] (du/dy)
  # operand is of shape [du1, ..., dun, dx1, ..., dxm] (du/dx)
  dims <- seq_len0(ndims(grad))
  nvl_dot_general(
    grad,
    operand,
    contracting_dims = list(dims, dims),
    batching_dims = list(integer(), integer())
  )
}

keep <- function(.required, ...) {
  funcs <- list(...)

  Map(
    function(f, id) {
      if (.required[[id]]) f
    },
    funcs,
    seq_along(.required)
  )
}

register_pullback_rule(p_add, function(primals, .required) {
  lhs <- primals[[1L]]
  rhs <- primals[[2L]]
  list(
    list(nvl_add(lhs, rhs)),
    function(grad) {
      keep(.required, \() grad, \() grad)
    }
  )
})

register_pullback_rule(p_mul, function(primals, .required) {
  lhs <- primals[[1L]]
  rhs <- primals[[2L]]
  list(
    list(nvl_mul(lhs, rhs)),
    function(grad) {
      keep(.required, \() nvl_mul(grad, rhs), \() nvl_mul(grad, lhs))
    }
  )
})

register_pullback_rule(p_sub, function(primals, .required) {
  lhs <- primals[[1L]]
  rhs <- primals[[2L]]
  list(
    list(nvl_sub(lhs, rhs)),
    function(grad) {
      keep(.required, \() grad, \() nvl_neg(grad))
    }
  )
})


register_pullback_rule(p_neg, function(primals, .required) {
  x <- primals[[1L]]
  list(
    list(nvl_neg(x)),
    function(grad) {
      keep(.required, \() nvl_neg(grad))
    }
  )
})

register_pullback_rule(p_transpose, function(primals, permutation, .required) {
  x <- primals[[1L]]
  list(
    list(nvl_transpose(x, permutation)),
    function(grad) {
      # TODO:
      .NotYetImplemented()
    }
  )
})

register_pullback_rule(
  p_dot_general,
  function(primals, contracting_dims, batching_dims, .required) {
    lhs <- primals[[1L]]
    rhs <- primals[[2L]]
    y <- nvl_dot_general(lhs, rhs, contracting_dims, batching_dims)
    list(
      list(y),
      function(grad) {
        # batching dimensions
        bd_lhs <- batching_dims[[1L]]
        bd_rhs <- batching_dims[[2L]]
        # contracting dimensions
        cd_lhs <- contracting_dims[[1L]]
        cd_rhs <- contracting_dims[[2L]]
        # remaining dimensions
        rem_dims <- function(operand, b_dims, c_dims) {
          ii <- c(b_dims, c_dims) + 1L
          seq_len0(ndims(operand))[if (length(ii)) -ii else TRUE]
        }
        rd_lhs <- rem_dims(lhs, bd_lhs, cd_lhs)
        rd_rhs <- rem_dims(rhs, bd_rhs, cd_rhs)

        # output dimensions
        bd_out <- seq_along0(bd_lhs)
        d_lhs_out <- seq_along0(rd_lhs) +
          if (length(bd_out)) bd_out[length(bd_out)] + 1L else 0L
        d_rhs_out <- seq_along0(rd_rhs) +
          if (length(d_lhs_out)) d_lhs_out[length(d_lhs_out)] + 1L else 0L

        conv_perm <- function(x) {
          # x is a permutation vector so that if we permute y by x
          # y[x[i]] = i
          # but stablehlo expects it so that:
          # y[i] = x[i]
          ids_new <- integer(length(x))
          for (i in seq_along(x)) {
            ids_new[x[i] + 1L] <- i - 1L
          }
          ids_new
        }

        cd_lhs2 <- cd_lhs[order(cd_rhs)]
        perm_lhs <- conv_perm(c(bd_lhs, rd_lhs, cd_lhs2))
        cd_rhs2 <- cd_rhs[order(cd_lhs)]
        perm_rhs <- conv_perm(c(bd_rhs, rd_rhs, cd_rhs2))

        make_grad_closure <- function(x, d_x_out, rd_x, bd_x, perm_y) {
          function() {
            grad_x <- nvl_dot_general(
              grad,
              x,
              contracting_dims = list(d_x_out, rd_x),
              batching_dims = list(bd_out, bd_x)
            )
            nvl_transpose(grad_x, perm_y)
          }
        }

        keep(
          .required,
          make_grad_closure(rhs, d_rhs_out, rd_rhs, bd_rhs, perm_lhs),
          make_grad_closure(lhs, d_lhs_out, rd_lhs, bd_lhs, perm_rhs)
        )
      }
    )
  }
)

register_pullback_rule(p_transpose, function(primals, permutation, .required) {
  x <- primals[[1L]]
  y <- nvl_transpose(x, permutation)
  list(
    list(y),
    function(grad) {
      # Inverse permutation: inv[p[i]] = i
      inv <- integer(length(permutation))
      for (i in seq_along(permutation)) {
        inv[permutation[[i]] + 1L] <- i - 1L
      }
      keep(.required, \() nvl_transpose(grad, inv))
    }
  )
})
