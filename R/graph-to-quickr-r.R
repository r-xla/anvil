#' @keywords internal
NULL

# Helpers ----------------------------------------------------------------------

quickr_tree_leaf_paths <- function(node, path = list()) {
  if (inherits(node, "LeafNode")) {
    return(list(path))
  }
  if (!inherits(node, "ListNode")) {
    cli_abort("Unsupported node type in tree: {.cls {class(node)[[1L]]}}")
  }

  out <- list()
  for (i in seq_along(node$nodes)) {
    nm <- if (is.null(node$names)) {
      ""
    } else {
      nm_i <- node$names[[i]]
      if (is.null(nm_i)) "" else nm_i
    }
    key <- if (nzchar(nm)) nm else i
    out <- c(out, quickr_tree_leaf_paths(node$nodes[[i]], c(path, list(key))))
  }
  out
}

quickr_paths_to_names <- function(paths) {
  raw <- vapply(
    paths,
    function(path) {
      if (!length(path)) {
        return("x")
      }
      parts <- vapply(
        path,
        function(p) {
          if (is.character(p)) p else as.character(p)
        },
        character(1L)
      )
      paste(parts, collapse = "_")
    },
    character(1L)
  )
  make.unique(make.names(raw))
}

quickr_make_const_arg_names <- function(user_arg_names, n) {
  n <- as.integer(n)
  if (n <= 0L) {
    return(character())
  }
  raw <- paste0("anvil_const", seq_len(n))
  combined <- make.unique(c(user_arg_names, raw))
  combined[(length(user_arg_names) + 1L):length(combined)]
}

quickr_dtype_to_r_ctor <- function(dt_chr) {
  if (dt_chr %in% c("f32", "f64")) {
    return("double")
  }
  if (grepl("^(u?i)(8|16|32|64)$", dt_chr)) {
    return("integer")
  }
  if (dt_chr %in% c("pred", "i1")) {
    return("logical")
  }
  cli_abort("Unsupported dtype for R code generation: {.val {dt_chr}}")
}

quickr_aval_to_declare_type_call <- function(aval) {
  dt_chr <- as.character(dtype(aval))
  ctor <- quickr_dtype_to_r_ctor(dt_chr)
  sh <- shape(aval)
  if (length(sh) > 5L) {
    cli_abort("quickr lowering currently supports tensors up to rank 5")
  }
  dims <- if (!length(sh)) 1L else sh
  as.call(c(list(as.name(ctor)), as.list(dims)))
}

quickr_declare_stmt <- function(arg_names, arg_avals) {
  type_calls <- Map(
    function(nm, aval) {
      arg <- list(quickr_aval_to_declare_type_call(aval))
      names(arg) <- nm
      as.call(c(list(as.name("type")), arg))
    },
    arg_names,
    arg_avals
  )
  as.call(c(list(as.name("declare")), type_calls))
}

quickr_scalar_cast <- function(value, dt_chr) {
  if (dt_chr %in% c("f32", "f64")) {
    return(as.double(value))
  }
  if (grepl("^(u?i)(8|16|32|64)$", dt_chr)) {
    return(as.integer(value))
  }
  if (dt_chr %in% c("pred", "i1")) {
    return(as.logical(value))
  }
  cli_abort("Unsupported dtype: {.val {dt_chr}}")
}

quickr_zero_literal_for <- function(aval) {
  dt_chr <- as.character(dtype(aval))
  if (dt_chr %in% c("pred", "i1")) {
    FALSE
  } else if (dt_chr %in% c("f32", "f64")) {
    0.0
  } else {
    0L
  }
}

quickr_emit_assign <- function(lhs_sym, rhs_expr) {
  list(rlang::call2("<-", lhs_sym, rhs_expr))
}

quickr_emit_full_like <- function(out_sym, value_expr, shape_out, out_aval) {
  if (!length(shape_out)) {
    return(quickr_emit_assign(out_sym, value_expr))
  }
  if (length(shape_out) > 5L) {
    cli_abort("constant/broadcast: only tensors up to rank 5 are currently supported")
  }

  quickr_emit_assign(
    out_sym,
    rlang::call2("array", value_expr, dim = as.integer(shape_out))
  )
}

quickr_expr_of_node <- function(node, node_expr) {
  if (is_graph_literal(node)) {
    return(quickr_scalar_cast(node@aval, as.character(node@dtype)))
  }
  expr <- node_expr[[node]]
  if (is.null(expr)) {
    cli_abort("Internal error: node not mapped to an expression")
  }
  expr
}


# Primitive emission ------------------------------------------------------------

quickr_emit_dot_general <- function(out_sym, lhs_expr, rhs_expr, lhs_shape, rhs_shape, out_shape, out_aval, contracting_dims, batching_dims) {
  lhs_shape <- as.integer(lhs_shape)
  rhs_shape <- as.integer(rhs_shape)
  out_shape <- as.integer(out_shape)

  lhs_rank <- length(lhs_shape)
  rhs_rank <- length(rhs_shape)
  out_rank <- length(out_shape)

  if (lhs_rank > 5L || rhs_rank > 5L || out_rank > 5L) {
    cli_abort("dot_general: only tensors up to rank 5 are supported")
  }

  cd_lhs <- as.integer(contracting_dims[[1L]])
  cd_rhs <- as.integer(contracting_dims[[2L]])
  bd_lhs <- as.integer(batching_dims[[1L]])
  bd_rhs <- as.integer(batching_dims[[2L]])

  if (length(cd_lhs) != length(cd_rhs)) {
    cli_abort("dot_general: contracting_dims must have the same length for lhs and rhs")
  }
  if (length(bd_lhs) != length(bd_rhs)) {
    cli_abort("dot_general: batching_dims must have the same length for lhs and rhs")
  }

  check_dims <- function(dims, rank, label) {
    if (!length(dims)) {
      return(invisible(NULL))
    }
    if (any(is.na(dims))) {
      cli_abort("dot_general: {label} must not contain missing values")
    }
    if (any(dims < 1L) || any(dims > rank)) {
      cli_abort("dot_general: {label} out of range for rank {rank}")
    }
    if (anyDuplicated(dims)) {
      cli_abort("dot_general: {label} must be unique")
    }
    invisible(NULL)
  }

  check_dims(cd_lhs, lhs_rank, "lhs contracting_dims")
  check_dims(cd_rhs, rhs_rank, "rhs contracting_dims")
  check_dims(bd_lhs, lhs_rank, "lhs batching_dims")
  check_dims(bd_rhs, rhs_rank, "rhs batching_dims")

  if (length(intersect(cd_lhs, bd_lhs))) {
    cli_abort("dot_general: lhs contracting_dims and batching_dims must be disjoint")
  }
  if (length(intersect(cd_rhs, bd_rhs))) {
    cli_abort("dot_general: rhs contracting_dims and batching_dims must be disjoint")
  }

  for (i in seq_along(bd_lhs)) {
    n_lhs <- lhs_shape[[bd_lhs[[i]]]]
    n_rhs <- rhs_shape[[bd_rhs[[i]]]]
    if (!identical(n_lhs, n_rhs)) {
      cli_abort("dot_general: batch dim sizes differ between lhs and rhs")
    }
  }
  for (i in seq_along(cd_lhs)) {
    n_lhs <- lhs_shape[[cd_lhs[[i]]]]
    n_rhs <- rhs_shape[[cd_rhs[[i]]]]
    if (!identical(n_lhs, n_rhs)) {
      cli_abort("dot_general: contracting dim sizes differ between lhs and rhs")
    }
  }

  free_lhs <- setdiff(seq_len(lhs_rank), c(bd_lhs, cd_lhs))
  free_rhs <- setdiff(seq_len(rhs_rank), c(bd_rhs, cd_rhs))
  expected_out <- c(lhs_shape[bd_lhs], lhs_shape[free_lhs], rhs_shape[free_rhs])
  if (!identical(out_shape, as.integer(expected_out))) {
    cli_abort("dot_general: output shape mismatch")
  }

  for_loop <- function(var_sym, upper, body) {
    as.call(list(
      as.name("for"),
      var_sym,
      rlang::call2("seq_len", as.integer(upper)),
      body
    ))
  }
  block <- function(...) as.call(c(list(as.name("{")), list(...)))
  subscript <- function(expr, idxs) {
    if (!length(idxs)) {
      return(expr)
    }
    as.call(c(list(as.name("["), expr), idxs))
  }

  nbatch <- length(bd_lhs)
  nfree_lhs <- length(free_lhs)

  out_idxs <- lapply(seq_len(out_rank), function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
  k_idxs <- lapply(seq_along(cd_lhs), function(i) as.name(paste0("k_", as.character(out_sym), "_", i)))
  acc <- as.name(paste0("acc_", as.character(out_sym)))

  lhs_idxs <- vector("list", lhs_rank)
  for (d in seq_len(lhs_rank)) {
    if (d %in% bd_lhs) {
      lhs_idxs[[d]] <- out_idxs[[match(d, bd_lhs)]]
    } else if (d %in% free_lhs) {
      lhs_idxs[[d]] <- out_idxs[[nbatch + match(d, free_lhs)]]
    } else {
      lhs_idxs[[d]] <- k_idxs[[match(d, cd_lhs)]]
    }
  }

  rhs_idxs <- vector("list", rhs_rank)
  for (d in seq_len(rhs_rank)) {
    if (d %in% bd_rhs) {
      rhs_idxs[[d]] <- out_idxs[[match(d, bd_rhs)]]
    } else if (d %in% free_rhs) {
      rhs_idxs[[d]] <- out_idxs[[nbatch + nfree_lhs + match(d, free_rhs)]]
    } else {
      rhs_idxs[[d]] <- k_idxs[[match(d, cd_rhs)]]
    }
  }

  lhs_at <- subscript(lhs_expr, lhs_idxs)
  rhs_at <- subscript(rhs_expr, rhs_idxs)
  zero <- quickr_zero_literal_for(out_aval)

  alloc_out <- function() {
    if (out_rank == 1L) {
      rlang::call2("array", zero, dim = out_shape[[1L]])
    } else if (out_rank == 2L) {
      rlang::call2("matrix", zero, nrow = out_shape[[1L]], ncol = out_shape[[2L]])
    } else {
      rlang::call2("array", zero, dim = out_shape)
    }
  }
  wrap_out_loops <- function(inner) {
    body <- inner
    for (d in seq_len(out_rank)) {
      body <- for_loop(out_idxs[[d]], out_shape[[d]], body)
    }
    body
  }

  if (!length(cd_lhs)) {
    if (out_rank == 0L) {
      return(quickr_emit_assign(out_sym, rlang::call2("*", lhs_at, rhs_at)))
    }

    alloc <- alloc_out()

    out_at <- subscript(out_sym, out_idxs)
    assign_elem <- rlang::call2("<-", out_at, rlang::call2("*", lhs_at, rhs_at))

    body <- wrap_out_loops(assign_elem)

    return(list(rlang::call2("<-", out_sym, alloc), body))
  }

  update_acc <- rlang::call2("<-", acc, rlang::call2("+", acc, rlang::call2("*", lhs_at, rhs_at)))
  contract_body <- update_acc
  for (i in seq_along(k_idxs)) {
    contract_body <- for_loop(k_idxs[[i]], lhs_shape[[cd_lhs[[i]]]], contract_body)
  }

  if (out_rank == 0L) {
    return(list(
      rlang::call2("<-", acc, zero),
      contract_body,
      rlang::call2("<-", out_sym, acc)
    ))
  }

  alloc <- alloc_out()

  out_at <- subscript(out_sym, out_idxs)
  elem_body <- block(
    rlang::call2("<-", acc, zero),
    contract_body,
    rlang::call2("<-", out_at, acc)
  )

  body <- wrap_out_loops(elem_body)

  list(rlang::call2("<-", out_sym, alloc), body)
}

quickr_emit_transpose <- function(out_sym, operand_expr, permutation, out_shape, out_aval) {
  if (length(out_shape) != 2L || length(permutation) != 2L) {
    cli_abort("transpose: only rank-2 tensors are supported")
  }
  if (identical(permutation, c(1L, 2L))) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }
  if (!identical(permutation, c(2L, 1L))) {
    cli_abort("transpose: only permutation=c(2,1) is supported")
  }

  zero <- quickr_zero_literal_for(out_aval)
  m <- as.integer(out_shape[[1L]])
  n <- as.integer(out_shape[[2L]])
  ii <- as.name(paste0("i_", as.character(out_sym)))
  jj <- as.name(paste0("j_", as.character(out_sym)))

  stmts <- quickr_emit_assign(out_sym, rlang::call2("matrix", zero, nrow = m, ncol = n))
  inner <- rlang::call2("<-", rlang::call2("[", out_sym, ii, jj), rlang::call2("[", operand_expr, jj, ii))

  c(
    stmts,
    list(as.call(list(
      as.name("for"),
      ii,
      rlang::call2("seq_len", m),
      as.call(list(as.name("for"), jj, rlang::call2("seq_len", n), inner))
    )))
  )
}

quickr_emit_broadcast_in_dim <- function(out_sym, operand_expr, shape_in, shape_out, broadcast_dimensions, out_aval) {
  broadcast_dimensions <- as.integer(broadcast_dimensions)
  if (identical(shape_in, shape_out)) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }

  rank_in <- length(shape_in)
  rank_out <- length(shape_out)

  if (rank_out == 0L) {
    cli_abort("broadcast_in_dim: cannot broadcast to a scalar")
  }

  if (rank_in == 0L) {
    return(quickr_emit_full_like(out_sym, operand_expr, shape_out, out_aval))
  }

  if (rank_in > 5L || rank_out > 5L) {
    cli_abort("broadcast_in_dim: only tensors up to rank 5 are supported")
  }
  if (length(broadcast_dimensions) != rank_in) {
    cli_abort("broadcast_in_dim: broadcast_dimensions must have length equal to rank(operand)")
  }
  if (any(broadcast_dimensions < 1L) || any(broadcast_dimensions > rank_out)) {
    cli_abort("broadcast_in_dim: broadcast_dimensions out of range")
  }
  if (anyDuplicated(broadcast_dimensions)) {
    cli_abort("broadcast_in_dim: broadcast_dimensions must be unique")
  }

  for (d_in in seq_len(rank_in)) {
    d_out <- broadcast_dimensions[[d_in]]
    n_in <- as.integer(shape_in[[d_in]])
    n_out <- as.integer(shape_out[[d_out]])
    if (!(n_in == 1L || identical(n_in, n_out))) {
      cli_abort("broadcast_in_dim: shape mismatch between input and output dimensions")
    }
  }

  out_zero <- quickr_zero_literal_for(out_aval)
  stmts <- quickr_emit_assign(out_sym, rlang::call2("array", out_zero, dim = as.integer(shape_out)))

  out_idxs <- lapply(seq_len(rank_out), function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))

  op_idxs <- vector("list", rank_in)
  for (d_in in seq_len(rank_in)) {
    d_out <- broadcast_dimensions[[d_in]]
    op_idxs[[d_in]] <- if (as.integer(shape_in[[d_in]]) == 1L) 1L else out_idxs[[d_out]]
  }

  op_at <- as.call(c(list(as.name("[")), list(operand_expr), op_idxs))
  out_at <- as.call(c(list(as.name("[")), list(out_sym), out_idxs))
  inner <- rlang::call2("<-", out_at, op_at)

  body <- inner
  # Iterate in column-major order (dim 1 varies fastest) for locality.
  for (d in seq_len(rank_out)) {
    body <- as.call(list(
      as.name("for"),
      out_idxs[[d]],
      rlang::call2("seq_len", as.integer(shape_out[[d]])),
      body
    ))
  }

  c(stmts, list(body))
}

quickr_emit_reduce_rank2_axis_loop <- function(out_sym, m, n, dims, drop, alloc_stmts, init_acc_expr, inner_start, update_builder) {
  dims <- as.integer(dims)
  ii <- as.name(paste0("i_", as.character(out_sym)))
  jj <- as.name(paste0("j_", as.character(out_sym)))
  acc <- as.name(paste0("acc_", as.character(out_sym)))

  outer_sym <- if (identical(dims, 2L)) ii else jj
  outer_n <- if (identical(dims, 2L)) m else n
  inner_sym <- if (identical(dims, 2L)) jj else ii
  inner_n <- if (identical(dims, 2L)) n else m

  assign_out <- if (identical(dims, 2L)) {
    if (isTRUE(drop)) {
      rlang::call2("<-", rlang::call2("[", out_sym, ii), acc)
    } else {
      rlang::call2("<-", rlang::call2("[", out_sym, ii, 1L), acc)
    }
  } else {
    if (isTRUE(drop)) {
      rlang::call2("<-", rlang::call2("[", out_sym, jj), acc)
    } else {
      rlang::call2("<-", rlang::call2("[", out_sym, 1L, jj), acc)
    }
  }

  body_parts <- list(rlang::call2("<-", acc, init_acc_expr))
  if (as.integer(inner_n) >= as.integer(inner_start)) {
    body_parts <- c(body_parts, list(as.call(list(
      as.name("for"),
      inner_sym,
      as.call(list(as.name(":"), as.integer(inner_start), as.integer(inner_n))),
      update_builder(ii, jj, acc)
    ))))
  }
  body_parts <- c(body_parts, list(assign_out))
  outer_body <- as.call(c(list(as.name("{")), body_parts))

  c(
    alloc_stmts,
    list(as.call(list(
      as.name("for"),
      outer_sym,
      rlang::call2("seq_len", as.integer(outer_n)),
      outer_body
    )))
  )
}

quickr_emit_reduce_sum <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
  dims <- sort(unique(as.integer(dims)))
  rank <- length(shape_in)
  zero <- quickr_zero_literal_for(out_aval)

  if (rank == 0L) {
    if (length(dims)) {
      cli_abort("sum: scalar reduction dims must be empty")
    }
    return(quickr_emit_assign(out_sym, operand_expr))
  }

  if (rank == 1L) {
    if (!length(dims)) {
      return(quickr_emit_assign(out_sym, operand_expr))
    }
    if (!identical(dims, 1L)) {
      cli_abort("sum: unsupported reduction dims for rank-1 tensor")
    }
    if (isTRUE(drop)) {
      return(quickr_emit_assign(out_sym, rlang::call2("sum", operand_expr)))
    }
    return(quickr_emit_assign(out_sym, rlang::call2("array", rlang::call2("sum", operand_expr), dim = 1L)))
  }

  if (rank == 2L) {
    m <- as.integer(shape_in[[1L]])
    n <- as.integer(shape_in[[2L]])
    ctor <- quickr_dtype_to_r_ctor(as.character(dtype(out_aval)))

    if (identical(dims, c(1L, 2L))) {
      if (isTRUE(drop)) {
        return(quickr_emit_assign(out_sym, rlang::call2("sum", operand_expr)))
      }
      return(quickr_emit_assign(out_sym, rlang::call2("matrix", rlang::call2("sum", operand_expr), nrow = 1L, ncol = 1L)))
    }

    update <- function(ii, jj, acc) rlang::call2("<-", acc, rlang::call2("+", acc, rlang::call2("[", operand_expr, ii, jj)))

    if (identical(dims, 2L)) {
      alloc <- if (isTRUE(drop)) {
        quickr_emit_assign(out_sym, rlang::call2(ctor, m))
      } else {
        quickr_emit_assign(out_sym, rlang::call2("matrix", zero, nrow = m, ncol = 1L))
      }
      return(quickr_emit_reduce_rank2_axis_loop(out_sym, m, n, 2L, drop, alloc, zero, 1L, update))
    }

    if (identical(dims, 1L)) {
      alloc <- if (isTRUE(drop)) {
        quickr_emit_assign(out_sym, rlang::call2(ctor, n))
      } else {
        quickr_emit_assign(out_sym, rlang::call2("matrix", zero, nrow = 1L, ncol = n))
      }
      return(quickr_emit_reduce_rank2_axis_loop(out_sym, m, n, 1L, drop, alloc, zero, 1L, update))
    }

    cli_abort("sum: unsupported reduction dims for rank-2 tensor")
  }

  if (!length(dims)) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }
  if (!identical(dims, seq_len(rank))) {
    cli_abort("sum: for rank > 2, only full reductions (dims = seq_len(rank)) are supported")
  }

  if (isTRUE(drop)) {
    return(quickr_emit_assign(out_sym, rlang::call2("sum", operand_expr)))
  }

  quickr_emit_assign(
    out_sym,
    rlang::call2("array", rlang::call2("sum", operand_expr), dim = rep(1L, rank))
  )
}

quickr_emit_reshape <- function(out_sym, operand_expr, shape_in, shape_out, out_aval) {
  shape_in <- as.integer(shape_in)
  shape_out <- as.integer(shape_out)

  if (length(shape_in) > 5L || length(shape_out) > 5L) {
    cli_abort("reshape: only tensors up to rank 5 are supported")
  }
  if (Reduce(`*`, shape_in, init = 1L) != Reduce(`*`, shape_out, init = 1L)) {
    cli_abort("reshape: input and output sizes differ")
  }

  nflat <- Reduce(`*`, shape_in, init = 1L)
  if (identical(as.integer(nflat), 1L)) {
    return(quickr_emit_full_like(out_sym, operand_expr, shape_out, out_aval))
  }

  ctor <- quickr_dtype_to_r_ctor(as.character(dtype(out_aval)))
  flat_sym <- as.name(paste0("flat_", as.character(out_sym)))
  idx_sym <- as.name(paste0("idx_", as.character(out_sym)))
  rank_in <- length(shape_in)
  rank_out <- length(shape_out)

  subscript <- function(expr, idxs) {
    if (!length(idxs)) {
      return(expr)
    }
    as.call(c(list(as.name("[")), list(expr), idxs))
  }
  row_major_loop <- function(idxs, shp, inner) {
    body <- inner
    if (!length(idxs)) {
      return(body)
    }
    for (d in rev(seq_along(idxs))) {
      body <- as.call(list(
        as.name("for"),
        idxs[[d]],
        rlang::call2("seq_len", as.integer(shp[[d]])),
        body
      ))
    }
    body
  }

  stmts <- list(
    rlang::call2("<-", flat_sym, rlang::call2(ctor, as.integer(nflat))),
    rlang::call2("<-", idx_sym, 0L)
  )

  if (rank_in == 0L) {
    stmts <- c(stmts, list(
      rlang::call2("<-", idx_sym, 1L),
      rlang::call2("<-", rlang::call2("[", flat_sym, 1L), operand_expr)
    ))
  } else {
    in_idxs <- lapply(seq_len(rank_in), function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
    elem_in <- subscript(operand_expr, in_idxs)
    inner_in <- as.call(c(
      list(as.name("{")),
      list(rlang::call2("<-", idx_sym, rlang::call2("+", idx_sym, 1L))),
      list(rlang::call2("<-", rlang::call2("[", flat_sym, idx_sym), elem_in))
    ))
    stmts <- c(stmts, list(row_major_loop(in_idxs, shape_in, inner_in)))
  }

  stmts <- c(stmts, list(rlang::call2("<-", idx_sym, 0L)))

  if (rank_out == 0L) {
    return(c(stmts, quickr_emit_assign(out_sym, rlang::call2("[", flat_sym, 1L))))
  }

  out_zero <- quickr_zero_literal_for(out_aval)
  alloc_out <- if (rank_out == 1L) {
    rlang::call2(ctor, as.integer(shape_out[[1L]]))
  } else if (rank_out == 2L) {
    rlang::call2("matrix", out_zero, nrow = as.integer(shape_out[[1L]]), ncol = as.integer(shape_out[[2L]]))
  } else {
    rlang::call2("array", out_zero, dim = shape_out)
  }

  stmts <- c(stmts, quickr_emit_assign(out_sym, alloc_out))

  out_idxs <- lapply(seq_len(rank_out), function(d) as.name(paste0("o_", as.character(out_sym), "_", d)))
  assign_out <- if (rank_out == 1L) {
    rlang::call2("<-", rlang::call2("[", out_sym, out_idxs[[1L]]), rlang::call2("[", flat_sym, idx_sym))
  } else {
    rlang::call2("<-", subscript(out_sym, out_idxs), rlang::call2("[", flat_sym, idx_sym))
  }
  inner_out <- as.call(c(
    list(as.name("{")),
    list(rlang::call2("<-", idx_sym, rlang::call2("+", idx_sym, 1L))),
    list(assign_out)
  ))
  stmts <- c(stmts, list(row_major_loop(out_idxs, shape_out, inner_out)))
  stmts
}


# Primitive lowering registry ---------------------------------------------------

quickr_register_prim_lowerer <- function(registry, name, fun) {
  for (nm in name) {
    registry[[nm]] <- fun
  }
  invisible(fun)
}

quickr_lower_registry <- local({
  reg <- new.env(parent = emptyenv())

  quickr_register_prim_lowerer(reg, "constant", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    out_sym <- out_syms[[1L]]
    out_aval <- out_avals[[1L]]
    dt_chr <- as.character(params$dtype)
    value_expr <- quickr_scalar_cast(params$value, dt_chr)
    quickr_emit_full_like(out_sym, value_expr, params$shape, out_aval)
  })

  quickr_register_prim_lowerer(reg, c("add", "sub", "mul", "divide"), function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    op <- switch(
      prim_name,
      add = "+",
      sub = "-",
      mul = "*",
      divide = "/",
      cli_abort("Internal error: unknown binary primitive: {.val {prim_name}}")
    )
    quickr_emit_assign(out_syms[[1L]], rlang::call2(op, inputs[[1L]], inputs[[2L]]))
  })

  quickr_register_prim_lowerer(reg, "negate", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    quickr_emit_assign(out_syms[[1L]], rlang::call2("-", inputs[[1L]]))
  })

  quickr_register_prim_lowerer(reg, "broadcast_in_dim", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    out_sym <- out_syms[[1L]]
    out_aval <- out_avals[[1L]]
    operand_node <- input_nodes[[1L]]
    if (!is_graph_value(operand_node)) {
      cli_abort("broadcast_in_dim: only GraphValue inputs are supported")
    }
    quickr_emit_broadcast_in_dim(
      out_sym,
      inputs[[1L]],
      shape(operand_node@aval),
      params$shape_out,
      params$broadcast_dimensions,
      out_aval
    )
  })

  quickr_register_prim_lowerer(reg, "dot_general", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    out_sym <- out_syms[[1L]]
    out_aval <- out_avals[[1L]]
    lhs_node <- input_nodes[[1L]]
    rhs_node <- input_nodes[[2L]]
    if (!is_graph_value(lhs_node) || !is_graph_value(rhs_node)) {
      cli_abort("dot_general: only GraphValue inputs are supported")
    }
    quickr_emit_dot_general(
      out_sym,
      inputs[[1L]],
      inputs[[2L]],
      shape(lhs_node@aval),
      shape(rhs_node@aval),
      shape(out_aval),
      out_aval,
      params$contracting_dims,
      params$batching_dims
    )
  })

  quickr_register_prim_lowerer(reg, "transpose", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    out_sym <- out_syms[[1L]]
    out_aval <- out_avals[[1L]]
    quickr_emit_transpose(out_sym, inputs[[1L]], params$permutation, shape(out_aval), out_aval)
  })

  quickr_register_prim_lowerer(reg, "reshape", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    out_sym <- out_syms[[1L]]
    out_aval <- out_avals[[1L]]
    operand_node <- input_nodes[[1L]]
    if (!is_graph_value(operand_node)) {
      cli_abort("reshape: only GraphValue inputs are supported")
    }
    quickr_emit_reshape(out_sym, inputs[[1L]], shape(operand_node@aval), params$shape, out_aval)
  })

  quickr_register_prim_lowerer(reg, "sum", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    out_sym <- out_syms[[1L]]
    out_aval <- out_avals[[1L]]
    operand_node <- input_nodes[[1L]]
    if (!is_graph_value(operand_node)) {
      cli_abort("sum: only GraphValue inputs are supported")
    }
    quickr_emit_reduce_sum(out_sym, inputs[[1L]], shape(operand_node@aval), params$dims, params$drop, out_aval)
  })

  reg
})


# Graph -> quickr-compatible R function ----------------------------------------

graph_to_quickr_r_function <- function(graph, include_declare = TRUE, pack_output = FALSE) {
  include_declare <- as.logical(include_declare)
  pack_output <- as.logical(pack_output)

  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls anvil::Graph}")
  }
  if (is.null(graph@in_tree) || is.null(graph@out_tree)) {
    cli_abort("{.arg graph} must have non-NULL {.field in_tree} and {.field out_tree}")
  }

  in_paths <- quickr_tree_leaf_paths(graph@in_tree)
  if (length(in_paths) != length(graph@inputs)) {
    cli_abort("Internal error: input tree size does not match graph inputs")
  }
  user_arg_names <- quickr_paths_to_names(in_paths)
  const_arg_names <- quickr_make_const_arg_names(user_arg_names, length(graph@constants))
  all_arg_names <- c(user_arg_names, const_arg_names)

  supported_prims <- ls(envir = quickr_lower_registry, all.names = TRUE)
  call_prims <- unique(vapply(graph@calls, \(x) x@primitive@name, character(1L)))
  unsupported_prims <- setdiff(call_prims, supported_prims)
  if (length(unsupported_prims)) {
    cli_abort(c(
      "{.fn graph_to_quickr_function} does not support these primitives: {toString(unsupported_prims)}",
      i = "Supported primitives: {toString(sort(supported_prims))}"
    ))
  }

  make_formals <- function(nms) {
    as.pairlist(stats::setNames(rep(list(quote(expr = )), length(nms)), nms))
  }

  node_expr <- hashtab()
  for (i in seq_along(graph@inputs)) {
    node_expr[[graph@inputs[[i]]]] <- as.name(user_arg_names[[i]])
  }

  stmts <- list()

  if (isTRUE(include_declare)) {
    arg_avals <- c(
      lapply(graph@inputs, \(x) x@aval),
      lapply(graph@constants, \(x) x@aval)
    )
    stmts <- c(stmts, list(quickr_declare_stmt(all_arg_names, arg_avals)))
  }

  for (i in seq_along(graph@constants)) {
    const_node <- graph@constants[[i]]
    if (!is_graph_value(const_node)) {
      cli_abort("quickr lowering: graph constants must be GraphValue nodes")
    }
    if (!is_concrete_tensor(const_node@aval)) {
      cli_abort("quickr lowering: graph constants must be concrete tensors")
    }
    node_expr[[const_node]] <- as.name(const_arg_names[[i]])
  }

  tmp_i <- 0L
  for (call in graph@calls) {
    input_exprs <- lapply(call@inputs, quickr_expr_of_node, node_expr = node_expr)
    out_syms <- vector("list", length(call@outputs))
    out_avals <- vector("list", length(call@outputs))
    for (i in seq_along(call@outputs)) {
      out_node <- call@outputs[[i]]
      if (!is_graph_value(out_node)) {
        cli_abort("Unsupported: non-GraphValue primitive outputs")
      }
      tmp_i <- tmp_i + 1L
      sym <- as.name(paste0("v", tmp_i))
      node_expr[[out_node]] <- sym
      out_syms[[i]] <- sym
      out_avals[[i]] <- out_node@aval
    }
    lower <- get0(call@primitive@name, envir = quickr_lower_registry, inherits = FALSE)
    stmts <- c(stmts, lower(call@primitive@name, input_exprs, call@params, out_syms, call@inputs, out_avals))
  }

  out_exprs <- lapply(graph@outputs, quickr_expr_of_node, node_expr = node_expr)

  if (isTRUE(pack_output)) {
    out_shapes <- lapply(graph@outputs, function(node) {
      if (is_graph_value(node)) {
        node@aval@shape@dims
      } else {
        integer()
      }
    })
    out_lens <- vapply(out_shapes, function(shp) {
      if (!length(shp)) 1L else Reduce(`*`, as.integer(shp), init = 1L)
    }, integer(1L))
    total_len <- sum(out_lens)

    out_sym <- as.name("out")
    idx_sym <- as.name("out_i")
    stmts <- c(stmts, list(rlang::call2("<-", out_sym, rlang::call2("double", as.integer(total_len)))))
    stmts <- c(stmts, list(rlang::call2("<-", idx_sym, 0L)))

    emit_pack_array <- function(src_expr, shp, tag) {
      rank <- length(shp)
      if (rank == 0L) {
        return(list(
          rlang::call2("<-", idx_sym, rlang::call2("+", idx_sym, 1L)),
          rlang::call2("<-", rlang::call2("[", out_sym, idx_sym), rlang::call2("as.double", src_expr))
        ))
      }

      idxs <- lapply(seq_len(rank), function(d) as.name(paste0("i_", tag, "_", d)))
      elem <- as.call(c(list(as.name("[")), list(src_expr), idxs))
      inner <- as.call(c(list(as.name("{")), c(
        list(rlang::call2("<-", idx_sym, rlang::call2("+", idx_sym, 1L))),
        list(rlang::call2("<-", rlang::call2("[", out_sym, idx_sym), rlang::call2("as.double", elem)))
      )))

      body <- inner
      for (d in seq_len(rank)) {
        body <- as.call(list(as.name("for"), idxs[[d]], rlang::call2("seq_len", as.integer(shp[[d]])), body))
      }
      list(body)
    }

    for (i in seq_along(out_exprs)) {
      tag <- paste0("out", i)
      stmts <- c(stmts, emit_pack_array(out_exprs[[i]], out_shapes[[i]], tag))
    }

    stmts <- c(stmts, list(out_sym))
  } else {
    if (!inherits(graph@out_tree, "LeafNode") || length(out_exprs) != 1L) {
      cli_abort("Internal error: {.arg pack_output} must be TRUE for graphs with multiple outputs")
    }
    stmts <- c(stmts, list(rlang::call2("<-", as.name("out"), out_exprs[[1L]]), as.name("out")))
  }

  f <- function() {
    NULL
  }
  formals(f) <- make_formals(all_arg_names)
  body(f) <- as.call(c(list(as.name("{")), stmts))

  if (isTRUE(include_declare) && !exists("declare", envir = baseenv(), inherits = FALSE)) {
    declare <- function(...) invisible(NULL)
  }
  f
}
