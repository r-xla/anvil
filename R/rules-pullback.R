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

register_pullback_rule(p_div, function(primals, .required) {
  lhs <- primals[[1L]]
  rhs <- primals[[2L]]
  # 1 / x = x^{-1} ->
  # ->
  browser()
  two <- nv_scalar(2, dtype(x))
  one <- nv_scalar(1, dtype(x))

  y <- nvl_div(lhs, rhs)
  browser()

  list(
    list(y),
    function(grad) {
      keep(.required, \() nvl_div(one, rhs), \() nvl_div(nvl_neg(y), rhs))
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

        # lhs has dimensions (bd_lhs, cd_lhs, rd_lhs), not necessarily in this order.
        # each "dimension" is a set of axes, not a single axis.
        # rhs has (bd_rhs, cd_rhs, rd_rhs), not necessarily in this order.
        # output has dimension (dims(bd_lhs), dims(rd_lhs), dims(rd_rhs))

        # the dim of grad_lhs is (dims(bd_lhs), dims(rd_lhs), dims(cd_lhs))
        # now, we transpose it to its original shape

        # bd_lhs indicates the position of the original batching dimensions,
        # the same for rd_lhs and cd_lhs.
        # to revert them to their original position

        # for grad_lhs we compute
        # grad[(bd_lhs, rd_lhs, rd_rhs)] * rhs[some_order(bd_rhs, rd_rhs, cd_rhs)]
        # (batch dims stay batch_dims, reducing dim becomes the rd_rhs)
        # we get the gradient back in the shape:
        # [bd_lhs, rd_lhs, some_order(cd_rhs)].
        # But we want it in its original order, so we need to permute it back.
        # If cd_rhs was [4, 2, 7], some_order would be cd_rhs[2], cd_rhs[1], cd_rhs[3]
        # The some_order is given my how

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

        # For grad_rhs we compute
        # grad[(bd_rhs, rd_rhs, rd_lhs)] * lhs[some_order(bd_lhs, rd_lhs, cd_lhs)]
        # (batch dims stay batch_dims, reducing dim becomes the rd_lhs)
        # We get the gradient back in the shape:
        # [bd_rhs, rd_rhs, cd_lhs].

        # But we want it in its original order, so we need to permute it back.

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

register_pullback_rule(p_reduce_sum, function(primals, dims, drop, .required) {
  operand <- primals[[1L]]
  # [a,b,c] -> [a,c] | [a,1,c]

  # the gradient we have has shape

  y <- nvl_reduce_sum(operand, dims, drop)
  list(
    list(y),
    function(grad) {
      keep(.required, \() {
        nv_broadcast_to(grad, shape(operand))
      })
    }
  )
})

register_pullback_rule(p_broadcast_in_dim, function(primals, shape_out, broadcast_dimensions) {
  operand <- primals[[1L]]
  y <- nvl_broadcast_in_dim(operand, shape_out, broadcast_dimensions)

  # grad: [a, b, c]
  # operand is [b, c]
  # -> How to get d_y/d_operand
  # probably we have to sum across those dimensions and then we might have to transpose
  # the result, because broadcast_dimensions might have screwed up the order

  list(
    list(y),
    function(grad) {
      keep(.required, \() {
        x <- nvl_reduce_sum(operand, dims = broadcast_dimensions, drop = TRUE)
        if (!ordered) {}
      })
    }
  )
})
