# Each pullback rule needs to define how to do the forward pass
# and backward pass
# Important: Don't use infix operators here, as they correspond to
# the broadcasted version

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
  two <- nv_scalar(2L, dtype(lhs))
  one <- nv_scalar(1L, dtype(lhs))

  y <- nvl_div(lhs, rhs)

  # note that we are using broadcasted functions here so we don't have to generate
  # 'one' and 'two' for the full shape
  list(
    list(y),
    function(grad) {
      keep(.required, \() nv_div(one, rhs), \() nv_div(nvl_neg(y), rhs))
    }
  )
})

register_pullback_rule(p_pow, function(primals, .required) {
  lhs <- primals[[1L]]
  rhs <- primals[[2L]]
  list(
    list(nvl_pow(lhs, rhs)),
    function(grad) {
      keep(
        .required,
        \() {
          nv_pow(lhs, rhs - nv_scalar(1, dtype(lhs)))
        },
        \() {
          # nee
          # TODO: Need log for this
          .NotYetImplemented()
        }
      )
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
          ii <- c(b_dims, c_dims)
          seq_len(ndims(operand))[if (length(ii)) -ii else TRUE]
        }
        rd_lhs <- rem_dims(lhs, bd_lhs, cd_lhs)
        rd_rhs <- rem_dims(rhs, bd_rhs, cd_rhs)

        # output dimensions
        bd_out <- seq_along(bd_lhs)
        d_lhs_out <- seq_along(rd_lhs) +
          if (length(bd_out)) bd_out[length(bd_out)] else 0L
        d_rhs_out <- seq_along(rd_rhs) +
          if (length(d_lhs_out)) d_lhs_out[length(d_lhs_out)] else 0L

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
            ids_new[x[i]] <- i
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
      inv <- integer(length(permutation))
      for (i in seq_along(permutation)) {
        inv[permutation[[i]]] <- i
      }
      keep(.required, \() nvl_transpose(grad, inv))
    }
  )
})

register_pullback_rule(p_reshape, function(primals, shape, .required) {
  operand <- primals[[1L]]
  y <- nvl_reshape(operand, shape)
  list(
    list(y),
    function(grad) {
      keep(.required, \() nvl_reshape(grad, shape(operand)))
    }
  )
})

register_pullback_rule(p_reduce_sum, function(primals, dims, drop, .required) {
  operand <- primals[[1L]]
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

register_pullback_rule(p_broadcast_in_dim, function(primals, shape_out, broadcast_dimensions, .required) {
  operand <- primals[[1L]]
  y <- nvl_broadcast_in_dim(operand, shape_out, broadcast_dimensions)

  list(
    list(y),
    function(grad) {
      keep(.required, \() {
        # Sum grad over the axes that were introduced by broadcasting
        # (i.e., all output axes not present in broadcast_dimensions)
        # we need to sum across dimensions that were added (not present in broadcast_dimensions)
        # as well as those that were expanded from 1 to the new dimension

        new_dims <- setdiff(seq_len(ndims(y)), broadcast_dimensions)
        expand_dims <- broadcast_dimensions[(shape(y)[broadcast_dimensions] != 1L) & (shape(operand) == 1L)]
        reduce_dims <- c(new_dims, expand_dims)

        x <- if (length(reduce_dims)) nvl_reduce_sum(grad, dims = reduce_dims, drop = FALSE) else grad

        # Drop the singlar added dimensions
        if (length(new_dims)) {
          reshape_dims <- shape(x)
          reshape_dims <- reshape_dims[-new_dims]
          x <- nvl_reshape(x, reshape_dims)
        }

        # If broadcast_dimensions are not in increasing order, reorder the
        # remaining axes back to the original operand axis order.
        if (is.unsorted(broadcast_dimensions)) {
          x <- nvl_transpose(x, order(broadcast_dimensions))
        }
        x
      })
    }
  )
})
