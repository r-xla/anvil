#' @include quickr.R
#' @keywords internal
NULL

#' Convert a Graph to an R function
#'
#' Converts (a supported subset of) `anvil::Graph` objects into a plain R
#' function that evaluates the same computation using base R operators.
#'
#' This is intended as a bridge to tools like {quickr}, which can consume R
#' functions.
#'
#' @param graph ([`Graph`])\cr
#'   Graph to convert.
#' @param include_declare (`logical(1)`)\cr
#'   Whether to include a `declare(type(...))` call at the top of the function
#'   body (useful for {quickr}). In plain R it is treated as a no-op. Default is `TRUE`.
#' @param pack_output (`logical(1)`)\cr
#'   Whether to force the function to return a single flat `double()` vector
#'   containing all output leaves (for {quickr} compatibility when the graph
#'   returns a nested output tree). Default is `FALSE`.
#' @return (`function`)
#' @export
graph_to_r_function <- function(graph, include_declare = TRUE, pack_output = FALSE) {
  include_declare <- as.logical(include_declare)
  pack_output <- as.logical(pack_output)

  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls anvil::Graph}")
  }
  if (is.null(graph@in_tree) || is.null(graph@out_tree)) {
    cli_abort("{.arg graph} must have non-NULL {.field in_tree} and {.field out_tree}")
  }

  .call2 <- function(op, ...) as.call(c(list(as.name(op)), list(...)))

  .tree_leaf_paths <- function(node, path = list()) {
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
      out <- c(out, .tree_leaf_paths(node$nodes[[i]], c(path, list(key))))
    }
    out
  }

  .paths_to_names <- function(paths) {
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

  .dtype_to_r_ctor <- function(dt_chr) {
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

  .declare_stmt <- NULL
  if (isTRUE(include_declare)) {
    .aval_to_declare_type_call <- function(aval) {
      dt_chr <- as.character(dtype(aval))
      ctor <- .dtype_to_r_ctor(dt_chr)
      sh <- shape(aval)
      dims <- if (!length(sh)) 1L else sh
      as.call(c(list(as.name(ctor)), as.list(dims)))
    }

    .declare_stmt <- function(arg_names, arg_avals) {
      type_calls <- Map(
        function(nm, aval) {
          arg <- list(.aval_to_declare_type_call(aval))
          names(arg) <- nm
          as.call(c(list(as.name("type")), arg))
        },
        arg_names,
        arg_avals
      )
      as.call(c(list(as.name("declare")), type_calls))
    }
  }

  .scalar_cast <- function(value, dt_chr) {
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

  .vector_literal <- function(vec) {
    stopifnot(is.atomic(vec))
    if (length(vec) == 0L) {
      cli_abort("Internal error: cannot build empty literal")
    }
    if (length(vec) == 1L) {
      return(vec[[1L]])
    }
    as.call(c(list(as.name("c")), as.list(vec)))
  }

  .zero_literal_for <- function(aval) {
    dt_chr <- as.character(dtype(aval))
    if (dt_chr %in% c("pred", "i1")) {
      FALSE
    } else if (dt_chr %in% c("f32", "f64")) {
      0.0
    } else {
      0L
    }
  }

  .emit_assign <- function(lhs_sym, rhs_expr) {
    list(.call2("<-", lhs_sym, rhs_expr))
  }

  .emit_sign_expr <- function(x_expr, out_aval) {
    dt_chr <- as.character(dtype(out_aval))
    ctor <- .dtype_to_r_ctor(dt_chr)

    one <- if (dt_chr %in% c("f32", "f64")) 1.0 else 1L
    neg_one <- if (dt_chr %in% c("f32", "f64")) -1.0 else -1L
    zero <- .zero_literal_for(out_aval)

    .call2(
      paste0("as.", ctor),
      .call2(
        "ifelse",
        .call2(">", x_expr, zero),
        one,
        .call2("ifelse", .call2("<", x_expr, zero), neg_one, zero)
      )
    )
  }

  .emit_atan2_expr <- function(y_expr, x_expr, out_aval) {
    # quickr doesn't support atan2(), but it does support atan() and ifelse().
    # Implement atan2(y, x) with quadrant correction using only supported ops.
    pi_val <- 3.1415926535897932384626433832795028841971693993751
    zero <- .zero_literal_for(out_aval)

    atan_yx <- .call2("atan", .call2("/", y_expr, x_expr))

    # if (x > 0) atan(y/x)
    # else if (x < 0 & y >= 0) atan(y/x) + pi
    # else if (x < 0 & y < 0)  atan(y/x) - pi
    # else if (x == 0 & y > 0)  pi/2
    # else if (x == 0 & y < 0) -pi/2
    # else 0
    .call2(
      "ifelse",
      .call2(">", x_expr, zero),
      atan_yx,
      .call2(
        "ifelse",
        .call2("&", .call2("<", x_expr, zero), .call2(">=", y_expr, zero)),
        .call2("+", atan_yx, pi_val),
        .call2(
          "ifelse",
          .call2("&", .call2("<", x_expr, zero), .call2("<", y_expr, zero)),
          .call2("-", atan_yx, pi_val),
          .call2(
            "ifelse",
            .call2("&", .call2("==", x_expr, zero), .call2(">", y_expr, zero)),
            .call2("/", pi_val, 2.0),
            .call2(
              "ifelse",
              .call2("&", .call2("==", x_expr, zero), .call2("<", y_expr, zero)),
              .call2("/", -pi_val, 2.0),
              zero
            )
          )
        )
      )
    )
  }

  .emit_dim_assign <- function(x_sym, dims) {
    list(.call2("<-", .call2("dim", x_sym), as.integer(dims)))
  }

  .emit_fill_flat <- function(x_sym, n, value_expr) {
    i_sym <- as.name(paste0("i_fill_", as.character(x_sym)))
    list(as.call(list(
      as.name("for"),
      i_sym,
      as.call(list(as.name(":"), 1L, as.integer(n))),
      .call2("<-", .call2("[", x_sym, i_sym), value_expr)
    )))
  }

  .emit_full_like <- function(out_sym, value_expr, shape_out, out_aval) {
    ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))
    if (!length(shape_out)) {
      return(.emit_assign(out_sym, value_expr))
    }
    if (length(shape_out) == 2L) {
      return(.emit_assign(
        out_sym,
        .call2("matrix", value_expr, nrow = as.integer(shape_out[[1L]]), ncol = as.integer(shape_out[[2L]]))
      ))
    }

    n <- Reduce(`*`, as.integer(shape_out), init = 1L)
    stmts <- .emit_assign(out_sym, .call2(ctor, n))
    if (length(shape_out) > 2L) {
      stmts <- c(stmts, .emit_dim_assign(out_sym, shape_out))
    }
    c(stmts, .emit_fill_flat(out_sym, n, value_expr))
  }

  .emit_inline_const <- function(out_sym, gval) {
    if (!is_graph_value(gval) || !is_concrete_tensor(gval@aval)) {
      cli_abort("Internal error: expected a concrete GraphValue constant")
    }
    arr <- as_array(gval@aval@data)
    shp <- shape(gval@aval)
    vec_expr <- .vector_literal(as.vector(arr))
    if (!length(shp)) {
      return(.emit_assign(out_sym, vec_expr))
    }
    if (length(shp) == 1L) {
      return(.emit_assign(out_sym, vec_expr))
    }
    if (length(shp) == 2L) {
      return(.emit_assign(out_sym, .call2("matrix", vec_expr, nrow = as.integer(shp[[1L]]), ncol = as.integer(shp[[2L]]))))
    }
    c(.emit_assign(out_sym, vec_expr), .emit_dim_assign(out_sym, shp))
  }

  .expr_of_node <- function(node, node_expr) {
    if (is_graph_literal(node)) {
      return(.scalar_cast(node@aval, as.character(node@dtype)))
    }
    expr <- node_expr[[node]]
    if (is.null(expr)) {
      cli_abort("Internal error: node not mapped to an expression")
    }
    expr
  }

  .emit_dot_general <- function(out_sym, lhs_expr, rhs_expr, lhs_shape, rhs_shape, out_shape, out_aval, contracting_dims, batching_dims) {
    rank <- length(out_shape)
    if (!(rank %in% c(2L, 3L))) {
      cli_abort("dot_general: only rank-2 and rank-3 outputs are supported")
    }

    .for_loop <- function(var_sym, upper, body) {
      as.call(list(
        as.name("for"),
        var_sym,
        as.call(list(as.name(":"), 1L, upper)),
        body
      ))
    }
    .block <- function(...) as.call(c(list(as.name("{")), list(...)))
    .subscript <- function(expr, idxs) as.call(c(list(as.name("["), expr), idxs))

    cd_lhs <- as.integer(contracting_dims[[1L]])
    cd_rhs <- as.integer(contracting_dims[[2L]])
    bd_lhs <- as.integer(batching_dims[[1L]])
    bd_rhs <- as.integer(batching_dims[[2L]])

    zero <- .zero_literal_for(out_aval)
    out_i <- as.name(paste0("i_", as.character(out_sym)))
    out_j <- as.name(paste0("j_", as.character(out_sym)))
    out_k <- as.name(paste0("k_", as.character(out_sym)))
    out_b <- as.name(paste0("b_", as.character(out_sym)))
    out_acc <- as.name(paste0("acc_", as.character(out_sym)))

    if (rank == 2L) {
      if (length(bd_lhs) || length(bd_rhs)) {
        cli_abort("dot_general: rank-2 output does not support batching dims")
      }
      if (!(length(cd_lhs) == 1L && length(cd_rhs) == 1L)) {
        cli_abort("dot_general: rank-2 output supports only one contracting dim")
      }
      cd_lhs <- cd_lhs[[1L]]
      cd_rhs <- cd_rhs[[1L]]
      if (!(cd_lhs %in% 1:2 && cd_rhs %in% 1:2)) {
        cli_abort("dot_general: unsupported contracting dims for rank-2 output")
      }

      rd_lhs <- setdiff(1:2, cd_lhs)
      rd_rhs <- setdiff(1:2, cd_rhs)
      if (!(length(rd_lhs) == 1L && length(rd_rhs) == 1L)) {
        cli_abort("dot_general: unsupported contracting dims for rank-2 output")
      }

      m <- as.integer(lhs_shape[[rd_lhs]])
      n <- as.integer(rhs_shape[[rd_rhs]])
      k <- as.integer(lhs_shape[[cd_lhs]])
      if (!identical(k, as.integer(rhs_shape[[cd_rhs]]))) {
        cli_abort("dot_general: contracting dims sizes differ")
      }
      if (!identical(as.integer(out_shape), c(m, n))) {
        cli_abort("dot_general: output shape mismatch")
      }

      idx_lhs <- vector("list", 2L)
      idx_rhs <- vector("list", 2L)
      idx_lhs[[rd_lhs]] <- out_i
      idx_lhs[[cd_lhs]] <- out_k
      idx_rhs[[rd_rhs]] <- out_j
      idx_rhs[[cd_rhs]] <- out_k

      lhs_at <- .subscript(lhs_expr, idx_lhs)
      rhs_at <- .subscript(rhs_expr, idx_rhs)

      update_acc <- .call2("<-", out_acc, .call2("+", out_acc, .call2("*", lhs_at, rhs_at)))

      inner <- .block(
        .emit_assign(out_acc, zero)[[1L]],
        .for_loop(out_k, k, update_acc),
        .call2("<-", .call2("[", out_sym, out_i, out_j), out_acc)
      )

      list(
        .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = n))[[1L]],
        .for_loop(out_i, m, .for_loop(out_j, n, inner))
      )
    } else {
      if (!(length(bd_lhs) == 1L && length(bd_rhs) == 1L && bd_lhs[[1L]] == 1L && bd_rhs[[1L]] == 1L)) {
        cli_abort("dot_general: rank-3 output supports only batching_dims=1")
      }
      if (!(length(cd_lhs) == 1L && length(cd_rhs) == 1L)) {
        cli_abort("dot_general: rank-3 output supports only one contracting dim")
      }
      cd_lhs <- cd_lhs[[1L]]
      cd_rhs <- cd_rhs[[1L]]
      if (!(cd_lhs %in% 2:3 && cd_rhs %in% 2:3)) {
        cli_abort("dot_general: unsupported contracting dims for rank-3 output")
      }

      rd_lhs <- setdiff(2:3, cd_lhs)
      rd_rhs <- setdiff(2:3, cd_rhs)
      if (!(length(rd_lhs) == 1L && length(rd_rhs) == 1L)) {
        cli_abort("dot_general: unsupported contracting dims for rank-3 output")
      }

      b <- as.integer(out_shape[[1L]])
      m <- as.integer(lhs_shape[[rd_lhs]])
      n <- as.integer(rhs_shape[[rd_rhs]])
      k <- as.integer(lhs_shape[[cd_lhs]])
      if (!identical(k, as.integer(rhs_shape[[cd_rhs]]))) {
        cli_abort("dot_general: contracting dims sizes differ")
      }
      if (!identical(as.integer(out_shape), c(b, m, n))) {
        cli_abort("dot_general: output shape mismatch")
      }

      ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))
      out_len <- as.integer(b * m * n)

      idx_lhs <- vector("list", 3L)
      idx_rhs <- vector("list", 3L)
      idx_lhs[[1L]] <- out_b
      idx_rhs[[1L]] <- out_b
      idx_lhs[[rd_lhs]] <- out_i
      idx_lhs[[cd_lhs]] <- out_k
      idx_rhs[[rd_rhs]] <- out_j
      idx_rhs[[cd_rhs]] <- out_k

      lhs_at <- .subscript(lhs_expr, idx_lhs)
      rhs_at <- .subscript(rhs_expr, idx_rhs)

      update_acc <- .call2("<-", out_acc, .call2("+", out_acc, .call2("*", lhs_at, rhs_at)))

      inner <- .block(
        .emit_assign(out_acc, zero)[[1L]],
        .for_loop(out_k, k, update_acc),
        .call2("<-", .call2("[", out_sym, out_b, out_i, out_j), out_acc)
      )

      c(
        .emit_assign(out_sym, .call2(ctor, out_len)),
        .emit_dim_assign(out_sym, c(b, m, n)),
        list(.for_loop(out_b, b, .for_loop(out_i, m, .for_loop(out_j, n, inner))))
      )
    }
  }

  .emit_transpose <- function(out_sym, operand_expr, permutation, out_shape, out_aval) {
    if (length(out_shape) != 2L || length(permutation) != 2L) {
      cli_abort("transpose: only rank-2 tensors are currently supported")
    }
    if (identical(permutation, c(1L, 2L))) {
      return(.emit_assign(out_sym, operand_expr))
    }
    if (!identical(permutation, c(2L, 1L))) {
      cli_abort("transpose: only permutation=c(2,1) is currently supported")
    }

    zero <- .zero_literal_for(out_aval)
    m <- as.integer(out_shape[[1L]])
    n <- as.integer(out_shape[[2L]])
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))

    stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = n))
    stmts <- c(stmts, list(as.call(list(
      as.name("for"), ii, as.call(list(as.name(":"), 1L, m)),
      as.call(list(
        as.name("for"), jj, as.call(list(as.name(":"), 1L, n)),
        .call2("<-",
          .call2("[", out_sym, ii, jj),
          .call2("[", operand_expr, jj, ii)
        )
      ))
    ))))
    stmts
  }

  .emit_elementwise_maxmin <- function(out_sym, lhs_expr, rhs_expr, out_shape, is_max, out_aval) {
    rank <- length(out_shape)
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))
    acc <- as.name(paste0("tmp_", as.character(out_sym)))

    if (rank == 0L) {
      cmp <- if (is_max) .call2(">", lhs_expr, rhs_expr) else .call2("<", lhs_expr, rhs_expr)
      return(.emit_assign(out_sym, as.call(list(as.name("if"), cmp, lhs_expr, rhs_expr))))
    }

    if (rank == 1L) {
      n <- as.integer(out_shape[[1L]])
      ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))
      stmts <- .emit_assign(out_sym, .call2(ctor, n))
      for_body <- {
        lhs_i <- .call2("[", lhs_expr, ii)
        rhs_i <- .call2("[", rhs_expr, ii)
        cmp <- if (is_max) .call2(">", lhs_i, rhs_i) else .call2("<", lhs_i, rhs_i)
        true_assign <- .call2("<-", .call2("[", out_sym, ii), lhs_i)
        false_assign <- .call2("<-", .call2("[", out_sym, ii), rhs_i)
        as.call(c(list(as.name("{")), list(as.call(list(as.name("if"), cmp, true_assign, false_assign)))))
      }
      c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, n)), for_body))))
    } else if (rank == 2L) {
      m <- as.integer(out_shape[[1L]])
      n <- as.integer(out_shape[[2L]])
      zero <- .zero_literal_for(out_aval)
      stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = n))
      inner_body <- {
        lhs_ij <- .call2("[", lhs_expr, ii, jj)
        rhs_ij <- .call2("[", rhs_expr, ii, jj)
        cmp <- if (is_max) .call2(">", lhs_ij, rhs_ij) else .call2("<", lhs_ij, rhs_ij)
        true_assign <- .call2("<-", .call2("[", out_sym, ii, jj), lhs_ij)
        false_assign <- .call2("<-", .call2("[", out_sym, ii, jj), rhs_ij)
        as.call(c(list(as.name("{")), list(as.call(list(as.name("if"), cmp, true_assign, false_assign)))))
      }
      c(
        stmts,
        list(as.call(list(
          as.name("for"),
          ii,
          as.call(list(as.name(":"), 1L, m)),
          as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), inner_body))
        )))
      )
    } else {
      cli_abort("maximum/minimum: only rank-0/1/2 tensors are currently supported")
    }
  }

  .emit_broadcast_in_dim <- function(out_sym, operand_expr, shape_in, shape_out, broadcast_dimensions, out_aval) {
    broadcast_dimensions <- as.integer(broadcast_dimensions)
    if (identical(shape_in, shape_out)) {
      return(.emit_assign(out_sym, operand_expr))
    }

    rank_in <- length(shape_in)
    rank_out <- length(shape_out)

    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))

    if (rank_out == 0L) {
      cli_abort("broadcast_in_dim: cannot broadcast to a scalar")
    }

    # Scalar -> anything
    if (rank_in == 0L) {
      return(.emit_full_like(out_sym, operand_expr, shape_out, out_aval))
    }

    # Rank-1 output
    if (rank_out == 1L) {
      if (!(rank_in == 1L && identical(broadcast_dimensions, 1L))) {
        cli_abort("broadcast_in_dim: unsupported rank-1 broadcast mapping")
      }
      n <- as.integer(shape_out[[1L]])
      ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))
      stmts <- .emit_assign(out_sym, .call2(ctor, n))
      idx <- if (as.integer(shape_in[[1L]]) == 1L) 1L else ii
      body <- .call2("<-", .call2("[", out_sym, ii), .call2("[", operand_expr, idx))
      c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, n)), body))))
    } else if (rank_out == 2L) {
      m <- as.integer(shape_out[[1L]])
      n <- as.integer(shape_out[[2L]])
      zero <- .zero_literal_for(out_aval)
      stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = n))

      if (rank_in == 1L) {
        if (identical(broadcast_dimensions, 1L)) {
          idx_i <- if (as.integer(shape_in[[1L]]) == 1L) 1L else ii
          inner <- .call2("<-", .call2("[", out_sym, ii, jj), .call2("[", operand_expr, idx_i))
        } else if (identical(broadcast_dimensions, 2L)) {
          idx_j <- if (as.integer(shape_in[[1L]]) == 1L) 1L else jj
          inner <- .call2("<-", .call2("[", out_sym, ii, jj), .call2("[", operand_expr, idx_j))
        } else {
          cli_abort("broadcast_in_dim: unsupported rank-2 broadcast mapping for rank-1 input")
        }
      } else if (rank_in == 2L) {
        if (!identical(broadcast_dimensions, c(1L, 2L))) {
          cli_abort("broadcast_in_dim: unsupported rank-2 broadcast mapping")
        }
        idx_i <- if (as.integer(shape_in[[1L]]) == 1L) 1L else ii
        idx_j <- if (as.integer(shape_in[[2L]]) == 1L) 1L else jj
        inner <- .call2("<-", .call2("[", out_sym, ii, jj), .call2("[", operand_expr, idx_i, idx_j))
      } else {
        cli_abort("broadcast_in_dim: only rank-0/1/2 inputs are currently supported")
      }

      c(
        stmts,
        list(as.call(list(
          as.name("for"),
          ii,
          as.call(list(as.name(":"), 1L, m)),
          as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), inner))
        )))
      )
    } else {
      cli_abort("broadcast_in_dim: only rank-1/2 outputs are currently supported")
    }
  }

  .emit_reduce_sum <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)
    zero <- .zero_literal_for(out_aval)
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))
    acc <- as.name(paste0("acc_", as.character(out_sym)))

    if (rank == 1L && identical(dims, 1L)) {
      n <- as.integer(shape_in[[1L]])
      if (isTRUE(drop)) {
        stmts <- list(.call2("<-", acc, zero))
        stmts <- c(stmts, list(as.call(list(
          as.name("for"),
          ii,
          as.call(list(as.name(":"), 1L, n)),
          .call2("<-", acc, .call2("+", acc, .call2("[", operand_expr, ii)))
        ))))
        c(stmts, .emit_assign(out_sym, acc))
      } else {
        stmts <- list(.call2("<-", acc, zero))
        stmts <- c(stmts, list(as.call(list(
          as.name("for"),
          ii,
          as.call(list(as.name(":"), 1L, n)),
          .call2("<-", acc, .call2("+", acc, .call2("[", operand_expr, ii)))
        ))))
        c(stmts, .emit_assign(out_sym, .call2("c", acc)))
      }
    } else if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])
      ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))

      if (identical(dims, c(1L, 2L))) {
        # Full reduction
        if (isTRUE(drop)) {
          return(.emit_assign(out_sym, .call2("sum", operand_expr)))
        }
        return(.emit_assign(out_sym, .call2("matrix", .call2("sum", operand_expr), nrow = 1L, ncol = 1L)))
      }

      if (identical(dims, 2L)) {
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, m))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, zero)),
            list(as.call(list(
              as.name("for"),
              jj,
              as.call(list(as.name(":"), 1L, n)),
              .call2("<-", acc, .call2("+", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, ii), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        } else {
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = 1L))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, zero)),
            list(as.call(list(
              as.name("for"),
              jj,
              as.call(list(as.name(":"), 1L, n)),
              .call2("<-", acc, .call2("+", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, ii, 1L), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        }
      } else if (identical(dims, 1L)) {
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, n))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, zero)),
            list(as.call(list(
              as.name("for"),
              ii,
              as.call(list(as.name(":"), 1L, m)),
              .call2("<-", acc, .call2("+", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, jj), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        } else {
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = 1L, ncol = n))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, zero)),
            list(as.call(list(
              as.name("for"),
              ii,
              as.call(list(as.name(":"), 1L, m)),
              .call2("<-", acc, .call2("+", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, 1L, jj), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        }
      } else {
        cli_abort("sum: unsupported reduction dims for rank-2 tensor")
      }
    } else {
      cli_abort("sum: only rank-1/2 tensors are currently supported")
    }
  }

  .emit_reduce_max <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))
    acc <- as.name(paste0("acc_", as.character(out_sym)))

    if (rank == 1L && identical(dims, 1L)) {
      n <- as.integer(shape_in[[1L]])
      init <- .call2("[", operand_expr, 1L)
      update <- as.call(list(
        as.name("if"),
        .call2(">", .call2("[", operand_expr, ii), acc),
        .call2("<-", acc, .call2("[", operand_expr, ii))
      ))
      stmts <- list(.call2("<-", acc, init))
      if (n > 1L) {
        stmts <- c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 2L, n)), update))))
      }
      if (isTRUE(drop)) {
        c(stmts, .emit_assign(out_sym, acc))
      } else {
        c(stmts, .emit_assign(out_sym, .call2("c", acc)))
      }
    } else if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])
      ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))

      if (identical(dims, c(1L, 2L))) {
        if (isTRUE(drop)) {
          return(.emit_assign(out_sym, .call2("max", operand_expr)))
        }
        return(.emit_assign(out_sym, .call2("matrix", .call2("max", operand_expr), nrow = 1L, ncol = 1L)))
      }

      if (identical(dims, 2L)) {
        zero <- .zero_literal_for(out_aval)
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, m))
          outer_body <- {
            init <- .call2("[", operand_expr, ii, 1L)
            update <- as.call(list(
              as.name("if"),
              .call2(">", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (n > 1L) list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 2L, n)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, ii), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        } else {
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = 1L))
          outer_body <- {
            init <- .call2("[", operand_expr, ii, 1L)
            update <- as.call(list(
              as.name("if"),
              .call2(">", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (n > 1L) list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 2L, n)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, ii, 1L), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        }
      } else if (identical(dims, 1L)) {
        zero <- .zero_literal_for(out_aval)
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, n))
          outer_body <- {
            init <- .call2("[", operand_expr, 1L, jj)
            update <- as.call(list(
              as.name("if"),
              .call2(">", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (m > 1L) list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 2L, m)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, jj), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        } else {
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = 1L, ncol = n))
          outer_body <- {
            init <- .call2("[", operand_expr, 1L, jj)
            update <- as.call(list(
              as.name("if"),
              .call2(">", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (m > 1L) list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 2L, m)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, 1L, jj), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        }
      } else {
        cli_abort("max: unsupported reduction dims for rank-2 tensor")
      }
    } else {
      cli_abort("max: only rank-1/2 tensors are currently supported")
    }
  }

  .emit_reduce_min <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))
    acc <- as.name(paste0("acc_", as.character(out_sym)))

    if (rank == 1L && identical(dims, 1L)) {
      n <- as.integer(shape_in[[1L]])
      init <- .call2("[", operand_expr, 1L)
      update <- as.call(list(
        as.name("if"),
        .call2("<", .call2("[", operand_expr, ii), acc),
        .call2("<-", acc, .call2("[", operand_expr, ii))
      ))
      stmts <- list(.call2("<-", acc, init))
      if (n > 1L) {
        stmts <- c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 2L, n)), update))))
      }
      if (isTRUE(drop)) {
        c(stmts, .emit_assign(out_sym, acc))
      } else {
        c(stmts, .emit_assign(out_sym, .call2("c", acc)))
      }
    } else if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])
      ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))

      if (identical(dims, c(1L, 2L))) {
        if (isTRUE(drop)) {
          return(.emit_assign(out_sym, .call2("min", operand_expr)))
        }
        return(.emit_assign(out_sym, .call2("matrix", .call2("min", operand_expr), nrow = 1L, ncol = 1L)))
      }

      if (identical(dims, 2L)) {
        zero <- .zero_literal_for(out_aval)
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, m))
          outer_body <- {
            init <- .call2("[", operand_expr, ii, 1L)
            update <- as.call(list(
              as.name("if"),
              .call2("<", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (n > 1L) list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 2L, n)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, ii), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        } else {
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = 1L))
          outer_body <- {
            init <- .call2("[", operand_expr, ii, 1L)
            update <- as.call(list(
              as.name("if"),
              .call2("<", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (n > 1L) list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 2L, n)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, ii, 1L), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        }
      } else if (identical(dims, 1L)) {
        zero <- .zero_literal_for(out_aval)
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, n))
          outer_body <- {
            init <- .call2("[", operand_expr, 1L, jj)
            update <- as.call(list(
              as.name("if"),
              .call2("<", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (m > 1L) list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 2L, m)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, jj), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        } else {
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = 1L, ncol = n))
          outer_body <- {
            init <- .call2("[", operand_expr, 1L, jj)
            update <- as.call(list(
              as.name("if"),
              .call2("<", .call2("[", operand_expr, ii, jj), acc),
              .call2("<-", acc, .call2("[", operand_expr, ii, jj))
            ))
            as.call(c(
              list(as.name("{")),
              list(.call2("<-", acc, init)),
              if (m > 1L) list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 2L, m)), update))) else list(),
              list(.call2("<-", .call2("[", out_sym, 1L, jj), acc))
            ))
          }
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        }
      } else {
        cli_abort("min: unsupported reduction dims for rank-2 tensor")
      }
    } else {
      cli_abort("min: only rank-1/2 tensors are currently supported")
    }
  }

  .emit_reduce_prod <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)
    one <- if (as.character(dtype(out_aval)) %in% c("f32", "f64")) 1.0 else 1L
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))
    acc <- as.name(paste0("acc_", as.character(out_sym)))

    if (rank == 1L && identical(dims, 1L)) {
      n <- as.integer(shape_in[[1L]])
      stmts <- list(.call2("<-", acc, one))
      stmts <- c(stmts, list(as.call(list(
        as.name("for"),
        ii,
        as.call(list(as.name(":"), 1L, n)),
        .call2("<-", acc, .call2("*", acc, .call2("[", operand_expr, ii)))
      ))))
      if (isTRUE(drop)) {
        c(stmts, .emit_assign(out_sym, acc))
      } else {
        c(stmts, .emit_assign(out_sym, .call2("c", acc)))
      }
    } else if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])
      ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))

      if (identical(dims, c(1L, 2L))) {
        if (isTRUE(drop)) {
          return(.emit_assign(out_sym, .call2("prod", operand_expr)))
        }
        return(.emit_assign(out_sym, .call2("matrix", .call2("prod", operand_expr), nrow = 1L, ncol = 1L)))
      }

      if (identical(dims, 2L)) {
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, m))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, one)),
            list(as.call(list(
              as.name("for"),
              jj,
              as.call(list(as.name(":"), 1L, n)),
              .call2("<-", acc, .call2("*", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, ii), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        } else {
          zero <- .zero_literal_for(out_aval)
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = m, ncol = 1L))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, one)),
            list(as.call(list(
              as.name("for"),
              jj,
              as.call(list(as.name(":"), 1L, n)),
              .call2("<-", acc, .call2("*", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, ii, 1L), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))))
        }
      } else if (identical(dims, 1L)) {
        if (isTRUE(drop)) {
          stmts <- .emit_assign(out_sym, .call2(ctor, n))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, one)),
            list(as.call(list(
              as.name("for"),
              ii,
              as.call(list(as.name(":"), 1L, m)),
              .call2("<-", acc, .call2("*", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, jj), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        } else {
          zero <- .zero_literal_for(out_aval)
          stmts <- .emit_assign(out_sym, .call2("matrix", zero, nrow = 1L, ncol = n))
          outer_body <- as.call(c(
            list(as.name("{")),
            list(.call2("<-", acc, one)),
            list(as.call(list(
              as.name("for"),
              ii,
              as.call(list(as.name(":"), 1L, m)),
              .call2("<-", acc, .call2("*", acc, .call2("[", operand_expr, ii, jj)))
            ))),
            list(.call2("<-", .call2("[", out_sym, 1L, jj), acc))
          ))
          c(stmts, list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))))
        }
      } else {
        cli_abort("prod: unsupported reduction dims for rank-2 tensor")
      }
    } else {
      cli_abort("prod: only rank-1/2 tensors are currently supported")
    }
  }

  .emit_reduce_any_quickr <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))
    acc <- as.name(paste0("acc_", as.character(out_sym)))
    op_int <- as.name(paste0("opint_", as.character(out_sym)))
    out_int <- as.name(paste0("outint_", as.character(out_sym)))

    # quickr can't index logical arrays (it renders them as (x/=0)), so we
    # convert to an integer 0/1 array first via ifelse() (-> Fortran merge()).
    op_int_expr <- if (rank == 2L) {
      .call2(
        "matrix",
        .call2("ifelse", operand_expr, 1L, 0L),
        nrow = as.integer(shape_in[[1L]]),
        ncol = as.integer(shape_in[[2L]])
      )
    } else {
      .call2("ifelse", operand_expr, 1L, 0L)
    }
    op_int_stmt <- .emit_assign(op_int, op_int_expr)

    if (rank == 1L && identical(dims, 1L)) {
      n <- as.integer(shape_in[[1L]])
      if (isTRUE(drop)) {
        return(c(
          op_int_stmt,
          .emit_assign(out_sym, .call2("!=", .call2("sum", op_int), 0L))
        ))
      }
      return(c(
        op_int_stmt,
        .emit_assign(out_sym, .call2("c", .call2("!=", .call2("sum", op_int), 0L)))
      ))
    }

    if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])

      if (identical(dims, c(1L, 2L))) {
        if (isTRUE(drop)) {
          return(c(op_int_stmt, .emit_assign(out_sym, .call2("!=", .call2("sum", op_int), 0L))))
        }
        return(c(
          op_int_stmt,
          .emit_assign(out_sym, .call2("matrix", .call2("!=", .call2("sum", op_int), 0L), nrow = 1L, ncol = 1L))
        ))
      }

      if (identical(dims, 2L)) {
        # row-wise any
        out_int_alloc <- if (isTRUE(drop)) {
          .emit_assign(out_int, .call2("integer", m))
        } else {
          .emit_assign(out_int, .call2("matrix", 0L, nrow = m, ncol = 1L))
        }
        outer_body <- as.call(c(
          list(as.name("{")),
          list(.call2("<-", acc, 0L)),
          list(as.call(list(
            as.name("for"),
            jj,
            as.call(list(as.name(":"), 1L, n)),
            .call2("<-", acc, .call2("+", acc, .call2("[", op_int, ii, jj)))
          ))),
          if (isTRUE(drop)) {
            list(.call2("<-", .call2("[", out_int, ii), acc))
          } else {
            list(.call2("<-", .call2("[", out_int, ii, 1L), acc))
          }
        ))
        return(c(
          op_int_stmt,
          out_int_alloc,
          list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))),
          .emit_assign(out_sym, .call2("!=", out_int, 0L))
        ))
      }

      if (identical(dims, 1L)) {
        # col-wise any
        out_int_alloc <- if (isTRUE(drop)) {
          .emit_assign(out_int, .call2("integer", n))
        } else {
          .emit_assign(out_int, .call2("matrix", 0L, nrow = 1L, ncol = n))
        }
        outer_body <- as.call(c(
          list(as.name("{")),
          list(.call2("<-", acc, 0L)),
          list(as.call(list(
            as.name("for"),
            ii,
            as.call(list(as.name(":"), 1L, m)),
            .call2("<-", acc, .call2("+", acc, .call2("[", op_int, ii, jj)))
          ))),
          if (isTRUE(drop)) {
            list(.call2("<-", .call2("[", out_int, jj), acc))
          } else {
            list(.call2("<-", .call2("[", out_int, 1L, jj), acc))
          }
        ))
        return(c(
          op_int_stmt,
          out_int_alloc,
          list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))),
          .emit_assign(out_sym, .call2("!=", out_int, 0L))
        ))
      }

      cli_abort("any: unsupported reduction dims for rank-2 tensor")
    }

    cli_abort("any: only rank-1/2 tensors are currently supported")
  }

  .emit_reduce_all_quickr <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)
    ii <- as.name(paste0("i_", as.character(out_sym)))
    jj <- as.name(paste0("j_", as.character(out_sym)))
    acc <- as.name(paste0("acc_", as.character(out_sym)))
    op_int <- as.name(paste0("opint_", as.character(out_sym)))
    out_int <- as.name(paste0("outint_", as.character(out_sym)))

    op_int_expr <- if (rank == 2L) {
      .call2(
        "matrix",
        .call2("ifelse", operand_expr, 1L, 0L),
        nrow = as.integer(shape_in[[1L]]),
        ncol = as.integer(shape_in[[2L]])
      )
    } else {
      .call2("ifelse", operand_expr, 1L, 0L)
    }
    op_int_stmt <- .emit_assign(op_int, op_int_expr)

    if (rank == 1L && identical(dims, 1L)) {
      n <- as.integer(shape_in[[1L]])
      if (isTRUE(drop)) {
        return(c(op_int_stmt, .emit_assign(out_sym, .call2("==", .call2("sum", op_int), n))))
      }
      return(c(op_int_stmt, .emit_assign(out_sym, .call2("c", .call2("==", .call2("sum", op_int), n)))))
    }

    if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])

      if (identical(dims, c(1L, 2L))) {
        if (isTRUE(drop)) {
          return(c(op_int_stmt, .emit_assign(out_sym, .call2("==", .call2("sum", op_int), as.integer(m * n)))))
        }
        return(c(
          op_int_stmt,
          .emit_assign(out_sym, .call2("matrix", .call2("==", .call2("sum", op_int), as.integer(m * n)), nrow = 1L, ncol = 1L))
        ))
      }

      if (identical(dims, 2L)) {
        out_int_alloc <- if (isTRUE(drop)) {
          .emit_assign(out_int, .call2("integer", m))
        } else {
          .emit_assign(out_int, .call2("matrix", 0L, nrow = m, ncol = 1L))
        }
        outer_body <- as.call(c(
          list(as.name("{")),
          list(.call2("<-", acc, 0L)),
          list(as.call(list(
            as.name("for"),
            jj,
            as.call(list(as.name(":"), 1L, n)),
            .call2("<-", acc, .call2("+", acc, .call2("[", op_int, ii, jj)))
          ))),
          if (isTRUE(drop)) {
            list(.call2("<-", .call2("[", out_int, ii), acc))
          } else {
            list(.call2("<-", .call2("[", out_int, ii, 1L), acc))
          }
        ))
        return(c(
          op_int_stmt,
          out_int_alloc,
          list(as.call(list(as.name("for"), ii, as.call(list(as.name(":"), 1L, m)), outer_body))),
          .emit_assign(out_sym, .call2("==", out_int, n))
        ))
      }

      if (identical(dims, 1L)) {
        out_int_alloc <- if (isTRUE(drop)) {
          .emit_assign(out_int, .call2("integer", n))
        } else {
          .emit_assign(out_int, .call2("matrix", 0L, nrow = 1L, ncol = n))
        }
        outer_body <- as.call(c(
          list(as.name("{")),
          list(.call2("<-", acc, 0L)),
          list(as.call(list(
            as.name("for"),
            ii,
            as.call(list(as.name(":"), 1L, m)),
            .call2("<-", acc, .call2("+", acc, .call2("[", op_int, ii, jj)))
          ))),
          if (isTRUE(drop)) {
            list(.call2("<-", .call2("[", out_int, jj), acc))
          } else {
            list(.call2("<-", .call2("[", out_int, 1L, jj), acc))
          }
        ))
        return(c(
          op_int_stmt,
          out_int_alloc,
          list(as.call(list(as.name("for"), jj, as.call(list(as.name(":"), 1L, n)), outer_body))),
          .emit_assign(out_sym, .call2("==", out_int, m))
        ))
      }

      cli_abort("all: unsupported reduction dims for rank-2 tensor")
    }

    cli_abort("all: only rank-1/2 tensors are currently supported")
  }

  .emit_reduce_any_base <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)

    if (rank == 1L && identical(dims, 1L)) {
      expr <- .call2("any", operand_expr)
      if (!isTRUE(drop)) {
        expr <- .call2("c", expr)
      }
      return(.emit_assign(out_sym, expr))
    }

    if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])

      if (identical(dims, c(1L, 2L))) {
        expr <- .call2("any", operand_expr)
        if (!isTRUE(drop)) {
          expr <- .call2("matrix", expr, nrow = 1L, ncol = 1L)
        }
        return(.emit_assign(out_sym, expr))
      }

      if (identical(dims, 2L)) {
        expr <- .call2("!=", .call2("rowSums", operand_expr), 0L)
        if (!isTRUE(drop)) {
          expr <- .call2("matrix", expr, nrow = m, ncol = 1L)
        }
        return(.emit_assign(out_sym, expr))
      }

      if (identical(dims, 1L)) {
        expr <- .call2("!=", .call2("colSums", operand_expr), 0L)
        if (!isTRUE(drop)) {
          expr <- .call2("matrix", expr, nrow = 1L, ncol = n)
        }
        return(.emit_assign(out_sym, expr))
      }

      cli_abort("any: unsupported reduction dims for rank-2 tensor")
    }

    cli_abort("any: only rank-1/2 tensors are currently supported")
  }

  .emit_reduce_all_base <- function(out_sym, operand_expr, shape_in, dims, drop, out_aval) {
    dims <- as.integer(dims)
    rank <- length(shape_in)

    if (rank == 1L && identical(dims, 1L)) {
      expr <- .call2("all", operand_expr)
      if (!isTRUE(drop)) {
        expr <- .call2("c", expr)
      }
      return(.emit_assign(out_sym, expr))
    }

    if (rank == 2L) {
      m <- as.integer(shape_in[[1L]])
      n <- as.integer(shape_in[[2L]])

      if (identical(dims, c(1L, 2L))) {
        expr <- .call2("all", operand_expr)
        if (!isTRUE(drop)) {
          expr <- .call2("matrix", expr, nrow = 1L, ncol = 1L)
        }
        return(.emit_assign(out_sym, expr))
      }

      if (identical(dims, 2L)) {
        expr <- .call2("==", .call2("rowSums", operand_expr), n)
        if (!isTRUE(drop)) {
          expr <- .call2("matrix", expr, nrow = m, ncol = 1L)
        }
        return(.emit_assign(out_sym, expr))
      }

      if (identical(dims, 1L)) {
        expr <- .call2("==", .call2("colSums", operand_expr), m)
        if (!isTRUE(drop)) {
          expr <- .call2("matrix", expr, nrow = 1L, ncol = n)
        }
        return(.emit_assign(out_sym, expr))
      }

      cli_abort("all: unsupported reduction dims for rank-2 tensor")
    }

    cli_abort("all: only rank-1/2 tensors are currently supported")
  }

  .emit_reduce_any <- if (isTRUE(include_declare)) .emit_reduce_any_quickr else .emit_reduce_any_base
  .emit_reduce_all <- if (isTRUE(include_declare)) .emit_reduce_all_quickr else .emit_reduce_all_base

  .emit_reshape <- function(out_sym, operand_expr, shape_in, shape_out, out_aval) {
    if (length(shape_in) > 2L || length(shape_out) > 2L) {
      cli_abort("reshape: only rank-0/1/2 tensors are currently supported")
    }
    if (Reduce(`*`, as.integer(shape_in), init = 1L) != Reduce(`*`, as.integer(shape_out), init = 1L)) {
      cli_abort("reshape: input and output sizes differ")
    }

    ctor <- .dtype_to_r_ctor(as.character(dtype(out_aval)))
    zero <- .zero_literal_for(out_aval)
    flat_sym <- as.name(paste0("flat_", as.character(out_sym)))
    idx_sym <- as.name(paste0("idx_", as.character(out_sym)))

    nflat <- Reduce(`*`, as.integer(shape_in), init = 1L)
    stmts <- list(.call2("<-", flat_sym, .call2(ctor, as.integer(nflat))))
    stmts <- c(stmts, list(.call2("<-", idx_sym, 0L)))

    # Fill flat in row-major order
    if (!length(shape_in)) {
      stmts <- c(stmts, list(.call2("<-", idx_sym, 1L), .call2("<-", .call2("[", flat_sym, 1L), operand_expr)))
    } else if (length(shape_in) == 1L) {
      ii <- as.name(paste0("i_", as.character(out_sym)))
      n1 <- as.integer(shape_in[[1L]])
      stmts <- c(stmts, list(as.call(list(
        as.name("for"), ii, as.call(list(as.name(":"), 1L, n1)),
        as.call(c(
          list(as.name("{")),
          list(.call2("<-", idx_sym, .call2("+", idx_sym, 1L))),
          list(.call2("<-", .call2("[", flat_sym, idx_sym), .call2("[", operand_expr, ii)))
        ))
      ))))
    } else {
      ii <- as.name(paste0("i_", as.character(out_sym)))
      jj <- as.name(paste0("j_", as.character(out_sym)))
      n1 <- as.integer(shape_in[[1L]])
      n2 <- as.integer(shape_in[[2L]])
      stmts <- c(stmts, list(as.call(list(
        as.name("for"), ii, as.call(list(as.name(":"), 1L, n1)),
        as.call(list(
          as.name("for"), jj, as.call(list(as.name(":"), 1L, n2)),
          as.call(c(
            list(as.name("{")),
            list(.call2("<-", idx_sym, .call2("+", idx_sym, 1L))),
            list(.call2("<-", .call2("[", flat_sym, idx_sym), .call2("[", operand_expr, ii, jj)))
          ))
        ))
      ))))
    }

    # Allocate output and fill in row-major order from flat
    stmts <- c(stmts, list(.call2("<-", idx_sym, 0L)))
    if (!length(shape_out)) {
      stmts <- c(stmts, .emit_assign(out_sym, .call2("[", flat_sym, 1L)))
      return(stmts)
    }
    if (length(shape_out) == 1L) {
      n1 <- as.integer(shape_out[[1L]])
      stmts <- c(stmts, .emit_assign(out_sym, .call2(ctor, n1)))
      ii <- as.name(paste0("o_", as.character(out_sym)))
      stmts <- c(stmts, list(as.call(list(
        as.name("for"), ii, as.call(list(as.name(":"), 1L, n1)),
        as.call(c(
          list(as.name("{")),
          list(.call2("<-", idx_sym, .call2("+", idx_sym, 1L))),
          list(.call2("<-", .call2("[", out_sym, ii), .call2("[", flat_sym, idx_sym)))
        ))
      ))))
      return(stmts)
    }

    n1 <- as.integer(shape_out[[1L]])
    n2 <- as.integer(shape_out[[2L]])
    stmts <- c(stmts, .emit_assign(out_sym, .call2("matrix", zero, nrow = n1, ncol = n2)))
    ii <- as.name(paste0("oi_", as.character(out_sym)))
    jj <- as.name(paste0("oj_", as.character(out_sym)))
    stmts <- c(stmts, list(as.call(list(
      as.name("for"), ii, as.call(list(as.name(":"), 1L, n1)),
      as.call(list(
        as.name("for"), jj, as.call(list(as.name(":"), 1L, n2)),
        as.call(c(
          list(as.name("{")),
          list(.call2("<-", idx_sym, .call2("+", idx_sym, 1L))),
          list(.call2("<-", .call2("[", out_sym, ii, jj), .call2("[", flat_sym, idx_sym)))
        ))
      ))
    ))))
    stmts
  }

  .emit_select <- function(out_sym, pred_expr, true_expr, false_expr, pred_shape, out_shape, out_aval) {
    rank <- length(out_shape)

    if (rank == 0L) {
      return(.emit_assign(out_sym, as.call(list(as.name("if"), pred_expr, true_expr, false_expr))))
    }

    # Use ifelse() so quickr can lower to Fortran merge() without indexing
    # a logical array (quickr currently cannot index logical arrays).
    if (rank == 1L) {
      return(.emit_assign(out_sym, .call2("ifelse", pred_expr, true_expr, false_expr)))
    }

    if (rank == 2L) {
      m <- as.integer(out_shape[[1L]])
      n <- as.integer(out_shape[[2L]])
      return(.emit_assign(
        out_sym,
        .call2("matrix", .call2("ifelse", pred_expr, true_expr, false_expr), nrow = m, ncol = n)
      ))
    }

    cli_abort("select: only rank-0/1/2 tensors are currently supported")
  }

  .make_slice <- function(target_sym, axis, axis_sym, rank) {
    target_nm <- as.character(target_sym)
    axis_nm <- as.character(axis_sym)
    idxs <- rep("", rank)
    idxs[[axis]] <- axis_nm
    str2lang(paste0(target_nm, "[", paste(idxs, collapse = ","), "]"))
  }

  .emit_mul_broadcast_axis <- function(out_sym, x_expr, y_expr, axis, shape_out) {
    rank <- length(shape_out)
    if (rank < 1L) {
      cli_abort("Internal error: cannot broadcast into rank-0 output")
    }
    axis <- as.integer(axis[[1L]])
    axis_size <- as.integer(shape_out[[axis]])
    k <- as.name(paste0("k_", as.character(out_sym), "_", axis))

    slice <- .make_slice(out_sym, axis, k, rank)
    rhs <- .call2("*", slice, .call2("[", y_expr, k))
    assign_slice <- .call2("<-", slice, rhs)

    list(
      .call2("<-", out_sym, x_expr),
      as.call(list(
        as.name("for"),
        k,
        .call2("seq_len", axis_size),
        assign_slice
      ))
    )
  }

  .register_prim_lowerer <- function(registry, name, fun) {
    for (nm in name) {
      registry[[nm]] <- fun
    }
    invisible(fun)
  }

  counter <- local({
    e <- new.env(parent = emptyenv())
    e$tmp <- 0L
    e$const <- 0L
    e
  })

  .lower_registry <- local({
    reg <- new.env(parent = emptyenv())

    .register_prim_lowerer(reg, "constant", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      dt_chr <- as.character(params$dtype)
      value <- params$value
      shp <- params$shape
      if (!checkmate::test_scalar(value)) {
        cli_abort("Only scalar `constant` values are supported")
      }
      value <- .scalar_cast(value, dt_chr)
      .emit_full_like(out_sym, value, shp, out_aval)
    })

    .register_prim_lowerer(reg, c("add", "sub", "mul", "divide", "power"), function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      op <- switch(
        prim_name,
        add = "+",
        sub = "-",
        mul = "*",
        divide = "/",
        power = "^",
        cli_abort("Internal error: unknown binary op primitive: {.val {prim_name}}")
      )
      .emit_assign(out_syms[[1L]], .call2(op, inputs[[1L]], inputs[[2L]]))
    })

    .register_prim_lowerer(reg, "mul_broadcast_axis", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      x_expr <- inputs[[1L]]
      y_expr <- inputs[[2L]]
      .emit_mul_broadcast_axis(out_sym, x_expr, y_expr, params$axis, params$shape_out)
    })

    .register_prim_lowerer(reg, "if", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      pred_node <- input_nodes[[1L]]
      if (is_graph_value(pred_node) && length(shape(pred_node@aval)) != 0L) {
        cli_abort("if: predicate must be a scalar (rank-0) boolean")
      }
      if (!is_graph(params$true_graph) || !is_graph(params$false_graph)) {
        cli_abort("if: expected {.field true_graph} and {.field false_graph} to be {.cls anvil::mut<Graph>}")
      }
      .emit_if(inputs[[1L]], params$true_graph, params$false_graph, out_syms)
    })

    .register_prim_lowerer(reg, "while", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      if (!is_graph(params$cond_graph) || !is_graph(params$body_graph)) {
        cli_abort("while: expected {.field cond_graph} and {.field body_graph} to be {.cls anvil::mut<Graph>}")
      }
      .emit_while(inputs, params$cond_graph, params$body_graph, out_syms)
    })

    .register_prim_lowerer(reg, c("negate", "abs", "exp", "sqrt", "log", "tanh", "tan", "floor", "ceil", "round"), function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      expr <- switch(
        prim_name,
        negate = .call2("-", inputs[[1L]]),
        abs = .call2("abs", inputs[[1L]]),
        exp = .call2("exp", inputs[[1L]]),
        sqrt = .call2("sqrt", inputs[[1L]]),
        log = .call2("log", inputs[[1L]]),
        tanh = .call2("tanh", inputs[[1L]]),
        tan = .call2("tan", inputs[[1L]]),
        floor = .call2("floor", inputs[[1L]]),
        ceil = .call2("ceiling", inputs[[1L]]),
        round = .call2("round", inputs[[1L]]),
        cli_abort("Internal error: unknown unary primitive: {.val {prim_name}}")
      )
      .emit_assign(out_syms[[1L]], expr)
    })

    .register_prim_lowerer(reg, "sign", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      .emit_assign(out_syms[[1L]], .emit_sign_expr(inputs[[1L]], out_avals[[1L]]))
    })

    .register_prim_lowerer(reg, "rsqrt", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      .emit_assign(out_syms[[1L]], .call2("/", 1.0, .call2("sqrt", inputs[[1L]])))
    })

    .register_prim_lowerer(reg, c("equal", "not_equal", "greater", "greater_equal", "less", "less_equal"), function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      op <- switch(
        prim_name,
        equal = "==",
        not_equal = "!=",
        greater = ">",
        greater_equal = ">=",
        less = "<",
        less_equal = "<=",
        cli_abort("Internal error: unknown comparison primitive: {.val {prim_name}}")
      )
      .emit_assign(out_syms[[1L]], .call2(op, inputs[[1L]], inputs[[2L]]))
    })

    .register_prim_lowerer(reg, "maximum", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      .emit_elementwise_maxmin(out_sym, inputs[[1L]], inputs[[2L]], shape(out_aval), TRUE, out_aval)
    })
    .register_prim_lowerer(reg, "minimum", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      .emit_elementwise_maxmin(out_sym, inputs[[1L]], inputs[[2L]], shape(out_aval), FALSE, out_aval)
    })

    .register_prim_lowerer(reg, "remainder", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      .emit_assign(out_syms[[1L]], .call2("%%", inputs[[1L]], inputs[[2L]]))
    })

    .register_prim_lowerer(reg, "not", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      .emit_assign(out_syms[[1L]], .call2("!", inputs[[1L]]))
    })
    .register_prim_lowerer(reg, c("and", "or"), function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      op <- switch(
        prim_name,
        and = "&",
        or = "|",
        cli_abort("Internal error: unknown boolean binary primitive: {.val {prim_name}}")
      )
      .emit_assign(out_syms[[1L]], .call2(op, inputs[[1L]], inputs[[2L]]))
    })
    .register_prim_lowerer(reg, "xor", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      .emit_assign(out_syms[[1L]], .call2("!=", inputs[[1L]], inputs[[2L]]))
    })

    .register_prim_lowerer(reg, c("shift_left", "shift_right_logical", "shift_right_arithmetic"), function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      if (prim_name == "shift_left") {
        .emit_assign(out_syms[[1L]], .call2("*", inputs[[1L]], .call2("^", 2L, inputs[[2L]])))
      } else {
        .emit_assign(out_syms[[1L]], .call2("%/%", inputs[[1L]], .call2("^", 2L, inputs[[2L]])))
      }
    })

    .register_prim_lowerer(reg, "atan2", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_aval <- out_avals[[1L]]
      .emit_assign(out_syms[[1L]], .emit_atan2_expr(inputs[[1L]], inputs[[2L]], out_aval))
    })

    .register_prim_lowerer(reg, "select", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      pred_node <- input_nodes[[1L]]
      pred_shape <- if (is_graph_value(pred_node)) shape(pred_node@aval) else integer()
      .emit_select(
        out_sym,
        inputs[[1L]],
        inputs[[2L]],
        inputs[[3L]],
        pred_shape = pred_shape,
        out_shape = shape(out_aval),
        out_aval = out_aval
      )
    })

    .register_prim_lowerer(reg, "convert", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      dt_chr <- as.character(params$dtype)
      conv <- .dtype_to_r_ctor(dt_chr)
      .emit_assign(out_syms[[1L]], .call2(paste0("as.", conv), inputs[[1L]]))
    })

    .register_prim_lowerer(reg, "broadcast_in_dim", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      shape_out <- params$shape_out
      broadcast_dimensions <- params$broadcast_dimensions
      operand_node <- input_nodes[[1L]]
      shape_in <- if (is_graph_value(operand_node)) shape(operand_node@aval) else integer()
      .emit_broadcast_in_dim(out_sym, inputs[[1L]], shape_in, shape_out, broadcast_dimensions, out_aval)
    })

    .register_prim_lowerer(reg, "dot_general", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      lhs_node <- input_nodes[[1L]]
      rhs_node <- input_nodes[[2L]]
      if (!is_graph_value(lhs_node) || !is_graph_value(rhs_node)) {
        cli_abort("dot_general: only GraphValue inputs are supported")
      }
      .emit_dot_general(
        out_sym,
        inputs[[1L]],
        inputs[[2L]],
        shape(lhs_node@aval),
        shape(rhs_node@aval),
        shape(out_aval),
        out_aval,
        contracting_dims = params$contracting_dims,
        batching_dims = params$batching_dims
      )
    })

    .register_prim_lowerer(reg, "transpose", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      .emit_transpose(out_sym, inputs[[1L]], params$permutation, shape(out_aval), out_aval)
    })

    .register_prim_lowerer(reg, "reshape", function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      if (!is_graph_value(operand_node)) {
        cli_abort("reshape: only GraphValue inputs are supported")
      }
      .emit_reshape(out_sym, inputs[[1L]], shape(operand_node@aval), params$shape, out_aval)
    })

    .register_prim_lowerer(reg, c("sum", "prod", "max", "min", "any", "all"), function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      if (!is_graph_value(operand_node)) {
        cli_abort("{.val {prim_name}}: only GraphValue inputs are supported")
      }
      switch(
        prim_name,
        sum = .emit_reduce_sum(out_sym, inputs[[1L]], shape(operand_node@aval), params$dims, params$drop, out_aval),
        prod = .emit_reduce_prod(out_sym, inputs[[1L]], shape(operand_node@aval), params$dims, params$drop, out_aval),
        max = .emit_reduce_max(out_sym, inputs[[1L]], shape(operand_node@aval), params$dims, params$drop, out_aval),
        min = .emit_reduce_min(out_sym, inputs[[1L]], shape(operand_node@aval), params$dims, params$drop, out_aval),
        any = .emit_reduce_any(out_sym, inputs[[1L]], shape(operand_node@aval), params$dims, params$drop, out_aval),
        all = .emit_reduce_all(out_sym, inputs[[1L]], shape(operand_node@aval), params$dims, params$drop, out_aval),
        cli_abort("Internal error: unknown reduction primitive: {.val {prim_name}}")
      )
    })

    reg
  })

  .emit_prim <- function(prim_name, inputs, params, out_syms, input_nodes, out_avals) {
    lower <- get0(prim_name, envir = .lower_registry, inherits = FALSE)
    if (is.null(lower)) {
      cli_abort("Unsupported primitive in graph_to_r_function(): {.val {prim_name}}")
    }
    lower(prim_name, inputs, params, out_syms, input_nodes, out_avals)
  }

  .inline_constants_for_graph <- function(graph, node_expr) {
    if (!length(graph@constants)) {
      return(list())
    }
    stmts <- list()
    for (gval in graph@constants) {
      if (!is.null(node_expr[[gval]])) {
        next
      }
      counter$const <- counter$const + 1L
      nm <- as.name(paste0("c_inline_", counter$const))
      node_expr[[gval]] <- nm
      stmts <- c(stmts, .emit_inline_const(nm, gval))
    }
    stmts
  }

  .ops_from_graph <- function(graph, node_expr) {
    supported_prims <- ls(envir = .lower_registry, all.names = TRUE)
    call_prims <- unique(vapply(graph@calls, \(x) x@primitive@name, character(1L)))
    unsupported_prims <- setdiff(call_prims, supported_prims)
    if (length(unsupported_prims)) {
      cli_abort(c(
        "graph_to_r_function() does not support these primitives: {toString(unsupported_prims)}",
        i = "Supported primitives: {toString(sort(supported_prims))}"
      ))
    }

    ops <- vector("list", length(graph@calls))
    op_i <- 0L
    for (call in graph@calls) {
      input_exprs <- lapply(call@inputs, .expr_of_node, node_expr = node_expr)
      out_syms <- vector("list", length(call@outputs))
      out_avals <- vector("list", length(call@outputs))
      for (i in seq_along(call@outputs)) {
        out_node <- call@outputs[[i]]
        if (!is_graph_value(out_node)) {
          cli_abort("Unsupported: non-GraphValue outputs")
        }
        counter$tmp <- counter$tmp + 1L
        sym <- as.name(paste0("v", counter$tmp))
        node_expr[[out_node]] <- sym
        out_syms[[i]] <- sym
        out_avals[[i]] <- out_node@aval
      }
      op_i <- op_i + 1L
      ops[[op_i]] <- list(
        prim_name = call@primitive@name,
        inputs = input_exprs,
        params = call@params,
        out_syms = out_syms,
        input_nodes = call@inputs,
        out_avals = out_avals
      )
    }
    ops
  }

  .emit_if <- function(pred_expr, true_graph, false_graph, out_syms) {
    emit_branch <- function(branch_graph) {
      if (length(branch_graph@outputs) != length(out_syms)) {
        cli_abort("if: branch output count does not match parent call outputs")
      }

      stmts <- .inline_constants_for_graph(branch_graph, node_expr)
      ops <- .ops_from_graph(branch_graph, node_expr)
      ops <- .fuse_broadcast_mul(ops)
      for (op in ops) {
        stmts <- c(stmts, .emit_prim(op$prim_name, op$inputs, op$params, op$out_syms, op$input_nodes, op$out_avals))
      }
      out_exprs <- lapply(branch_graph@outputs, .expr_of_node, node_expr = node_expr)
      for (i in seq_along(out_syms)) {
        stmts <- c(stmts, .emit_assign(out_syms[[i]], out_exprs[[i]]))
      }
      as.call(c(list(as.name("{")), stmts))
    }

    as.call(list(as.name("if"), pred_expr, emit_branch(true_graph), emit_branch(false_graph)))
  }

  .emit_while <- function(init_exprs, cond_graph, body_graph, state_syms) {
    if (length(cond_graph@outputs) != 1L) {
      cli_abort("while: cond_graph must have exactly one output")
    }
    cond_out <- cond_graph@outputs[[1L]]
    if (!is_graph_value(cond_out) || !(as.character(dtype(cond_out@aval)) %in% c("pred", "i1")) || length(shape(cond_out@aval)) != 0L) {
      cli_abort("while: cond_graph output must be a scalar boolean")
    }

    if (length(body_graph@outputs) != length(state_syms)) {
      cli_abort("while: body_graph output count does not match parent call outputs")
    }
    if (length(cond_graph@inputs) != length(state_syms) || length(body_graph@inputs) != length(state_syms)) {
      cli_abort("while: cond_graph/body_graph input count does not match parent call inputs")
    }

    # Hoist constants used by cond/body graphs outside the loop
    stmts <- .inline_constants_for_graph(cond_graph, node_expr)
    stmts <- c(stmts, .inline_constants_for_graph(body_graph, node_expr))

    # Initialize state from the primitive's inputs
    for (i in seq_along(state_syms)) {
      stmts <- c(stmts, .emit_assign(state_syms[[i]], init_exprs[[i]]))
    }

    # Build per-iteration condition block: { <cond ops>; <cond_expr> }
    node_expr_cond <- hashtab()
    for (i in seq_along(state_syms)) {
      node_expr_cond[[cond_graph@inputs[[i]]]] <- state_syms[[i]]
    }
    for (gval in cond_graph@constants) {
      node_expr_cond[[gval]] <- .expr_of_node(gval, node_expr = node_expr)
    }
    cond_ops <- .ops_from_graph(cond_graph, node_expr_cond)
    cond_ops <- .fuse_broadcast_mul(cond_ops)
    cond_stmts <- list()
    for (op in cond_ops) {
      cond_stmts <- c(cond_stmts, .emit_prim(op$prim_name, op$inputs, op$params, op$out_syms, op$input_nodes, op$out_avals))
    }
    cond_sym <- .expr_of_node(cond_out, node_expr = node_expr_cond)

    # Build per-iteration body block: compute new state then assign simultaneously
    node_expr_body <- hashtab()
    for (i in seq_along(state_syms)) {
      node_expr_body[[body_graph@inputs[[i]]]] <- state_syms[[i]]
    }
    for (gval in body_graph@constants) {
      node_expr_body[[gval]] <- .expr_of_node(gval, node_expr = node_expr)
    }
    body_ops <- .ops_from_graph(body_graph, node_expr_body)
    body_ops <- .fuse_broadcast_mul(body_ops)
    body_stmts <- list()
    for (op in body_ops) {
      body_stmts <- c(body_stmts, .emit_prim(op$prim_name, op$inputs, op$params, op$out_syms, op$input_nodes, op$out_avals))
    }

    tmp_syms <- vector("list", length(state_syms))
    for (i in seq_along(state_syms)) {
      counter$tmp <- counter$tmp + 1L
      tmp_syms[[i]] <- as.name(paste0("v", counter$tmp))
    }
    body_out_exprs <- lapply(body_graph@outputs, .expr_of_node, node_expr = node_expr_body)
    for (i in seq_along(tmp_syms)) {
      body_stmts <- c(body_stmts, .emit_assign(tmp_syms[[i]], body_out_exprs[[i]]))
    }
    for (i in seq_along(state_syms)) {
      body_stmts <- c(body_stmts, .emit_assign(state_syms[[i]], tmp_syms[[i]]))
    }

    body_block <- as.call(c(list(as.name("{")), c(body_stmts, cond_stmts)))

    c(stmts, cond_stmts, list(as.call(list(as.name("while"), cond_sym, body_block))))
  }

  .build_output_expr <- function(node, leaves) {
    if (inherits(node, "LeafNode")) {
      return(leaves[[node$i]])
    }
    if (!inherits(node, "ListNode")) {
      cli_abort("Unsupported output tree node type: {.cls {class(node)[[1L]]}}")
    }
    args <- lapply(node$nodes, .build_output_expr, leaves = leaves)
    if (!is.null(node$names)) {
      names(args) <- node$names
    }
    as.call(c(list(as.name("list")), args))
  }

  in_paths <- .tree_leaf_paths(graph@in_tree)
  if (length(in_paths) != length(graph@inputs)) {
    cli_abort("Internal error: input tree size does not match graph inputs")
  }
  input_arg_names <- .paths_to_names(in_paths)

  arg_names <- input_arg_names

  const_names <- if (length(graph@constants)) {
    raw <- paste0("const", seq_along(graph@constants))
    combined <- make.unique(c(arg_names, raw))
    combined[(length(arg_names) + 1L):length(combined)]
  } else {
    character()
  }

  make_formals <- function(nms) {
    as.pairlist(stats::setNames(rep(list(quote(expr = )), length(nms)), nms))
  }

  node_expr <- hashtab()
  for (i in seq_along(graph@inputs)) {
    node_expr[[graph@inputs[[i]]]] <- as.name(input_arg_names[[i]])
  }

  stmts <- list()

  if (isTRUE(include_declare)) {
    avs <- lapply(graph@inputs, \(x) x@aval)
    stmts <- c(stmts, list(.declare_stmt(arg_names, avs)))
  }

  if (length(graph@constants)) {
    for (i in seq_along(graph@constants)) {
      nm <- const_names[[i]]
      node_expr[[graph@constants[[i]]]] <- as.name(nm)
      stmts <- c(stmts, .emit_inline_const(as.name(nm), graph@constants[[i]]))
    }
  }

  counter$tmp <- 0L
  counter$const <- 0L
  ops <- .ops_from_graph(graph, node_expr)

  .fuse_broadcast_mul <- function(ops) {
    is_same_sym <- function(expr, sym) {
      is.name(expr) && identical(expr, sym)
    }

    count_sym_uses <- function(ops, sym) {
      n <- 0L
      for (op in ops) {
        for (inp in op$inputs) {
          if (is_same_sym(inp, sym)) n <- n + 1L
        }
      }
      n
    }

    out <- list()
    i <- 1L
    n <- length(ops)
    while (i <= n) {
      op <- ops[[i]]
      if (op$prim_name == "broadcast_in_dim" && i < n && length(op$out_syms) == 1L) {
        b_sym <- op$out_syms[[1L]]
        if (count_sym_uses(ops, b_sym) == 1L) {
          next_op <- ops[[i + 1L]]
          if (next_op$prim_name == "mul" && length(next_op$out_syms) == 1L) {
            b_pos <- which(vapply(next_op$inputs, is_same_sym, logical(1L), sym = b_sym))
            if (length(b_pos) == 1L) {
              other_pos <- if (b_pos == 1L) 2L else 1L

              y_node <- op$input_nodes[[1L]]
              if (is_graph_value(y_node) && length(shape(y_node@aval)) == 1L) {
                shape_out <- op$params$shape_out
                bd <- as.integer(op$params$broadcast_dimensions)
                if (length(bd) == 1L) {
                  axis <- bd[[1L]]
                  ok_axis <- axis >= 1L && axis <= length(shape_out)
                  ok_len <- as.integer(shape_out[[axis]]) == as.integer(shape(y_node@aval)[[1L]])

                  x_node <- next_op$input_nodes[[other_pos]]
                  if (ok_axis && ok_len && is_graph_value(x_node) && identical(shape(x_node@aval), shape_out)) {
                    fused <- list(
                      prim_name = "mul_broadcast_axis",
                      inputs = list(next_op$inputs[[other_pos]], op$inputs[[1L]]),
                      params = list(axis = axis, shape_out = shape_out),
                      out_syms = next_op$out_syms,
                      input_nodes = next_op$input_nodes,
                      out_avals = next_op$out_avals
                    )
                    out <- c(out, list(fused))
                    i <- i + 2L
                    next
                  }
                }
              }
            }
          }
        }
      }
      out <- c(out, list(op))
      i <- i + 1L
    }
    out
  }

  ops <- .fuse_broadcast_mul(ops)
  for (op in ops) {
    stmts <- c(stmts, .emit_prim(op$prim_name, op$inputs, op$params, op$out_syms, op$input_nodes, op$out_avals))
  }

  out_leaves <- lapply(graph@outputs, .expr_of_node, node_expr = node_expr)

  if (isTRUE(pack_output)) {
    out_nodes <- graph@outputs
    out_shapes <- lapply(out_nodes, function(node) {
      if (is_graph_value(node)) {
        shape(node@aval)
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
    stmts <- c(stmts, list(.call2("<-", out_sym, .call2("double", as.integer(total_len)))))
    stmts <- c(stmts, list(.call2("<-", idx_sym, 0L)))

    emit_pack_scalar <- function(src_expr) {
      list(
        .call2("<-", idx_sym, .call2("+", idx_sym, 1L)),
        .call2("<-", .call2("[", out_sym, idx_sym), .call2("as.double", src_expr))
      )
    }

    emit_pack_array <- function(src_expr, shp, tag) {
      rank <- length(shp)
      if (rank == 0L) {
        return(emit_pack_scalar(src_expr))
      }
      idxs <- lapply(seq_len(rank), function(d) as.name(paste0("i_", tag, "_", d)))

      elem <- as.call(c(list(as.name("[")), list(src_expr), idxs))
      inner <- as.call(c(list(as.name("{")), c(
        list(.call2("<-", idx_sym, .call2("+", idx_sym, 1L))),
        list(.call2("<-", .call2("[", out_sym, idx_sym), .call2("as.double", elem)))
      )))

      body <- inner
      for (d in seq_len(rank)) {
        body <- as.call(list(
          as.name("for"),
          idxs[[d]],
          .call2("seq_len", as.integer(shp[[d]])),
          body
        ))
      }
      list(body)
    }

    for (i in seq_along(out_leaves)) {
      tag <- paste0("out", i)
      shp <- out_shapes[[i]]
      stmts <- c(stmts, emit_pack_array(out_leaves[[i]], shp, tag))
    }

    stmts <- c(stmts, list(out_sym))
  } else {
    out_expr <- .build_output_expr(graph@out_tree, out_leaves)
    stmts <- c(stmts, list(.call2("<-", as.name("out"), out_expr), as.name("out")))
  }
  body_expr <- as.call(c(list(as.name("{")), stmts))

  f <- function() {
    NULL
  }
  formals(f) <- make_formals(arg_names)
  body(f) <- body_expr
  if (isTRUE(include_declare) && !exists("declare", envir = baseenv(), inherits = FALSE)) {
    declare <- function(...) invisible(NULL)
  }
  f
}
