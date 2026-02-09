test_that("quickr pipeline matches PJRT: core math + comparisons + reductions", {
  skip_if_no_quickr_or_pjrt()

  core_ops <- function(x_f64, y_f64, x_i32, y_i32, p, q) {
    a <- x_f64 + y_f64
    b <- x_f64 - y_f64
    c <- x_f64 * y_f64
    d <- x_f64 / (y_f64 + 1)
    e <- nv_negate(x_f64)

    ax <- nv_abs(x_f64)
    sq <- nv_sqrt(ax + 0.1)
    lg <- nv_log(ax + 1.1)
    ex <- nv_exp(x_f64)
    em1 <- nv_expm1(x_f64)
    l1p <- nv_log1p(ax * 0.1)
    lo <- nv_logistic(x_f64)
    si <- nv_sine(x_f64)
    co <- nv_cosine(x_f64)
    ta <- nv_tan(x_f64 * 0.1)
    th <- nv_tanh(x_f64)
    fl <- nv_floor(x_f64)
    ce <- nv_ceil(x_f64)

    mx <- nv_max(x_f64, y_f64)
    mn <- nv_min(x_f64, y_f64)
    pw <- nv_pow(ax + 0.1, nv_abs(y_f64) + 0.1)

    eq <- nv_eq(x_i32, y_i32)
    ne <- nv_ne(x_i32, y_i32)
    gt <- nv_gt(x_i32, y_i32)
    ge <- nv_ge(x_i32, y_i32)
    lt <- nv_lt(x_i32, y_i32)
    le <- nv_le(x_i32, y_i32)

    aand <- nv_and(p, q)
    oor <- nv_or(p, q)
    xxor <- nv_xor(p, q)
    nn <- nv_not(p)

    sel <- nv_ifelse(eq, x_f64, y_f64)

    sm <- sum(x_f64)
    rs1 <- nv_reduce_sum(x_f64, dims = 1L, drop = TRUE)
    rs2 <- nv_reduce_sum(x_f64, dims = 2L, drop = FALSE)
    rp <- nv_reduce_prod(ax + 1.0, dims = 2L, drop = TRUE)
    ra <- nv_reduce_any(eq, dims = 1L, drop = TRUE)
    rall <- nv_reduce_all(eq, dims = c(1L, 2L), drop = TRUE)

    rmax <- nv_reduce_max(x_f64, dims = 2L, drop = TRUE)
    rmin <- nv_reduce_min(x_f64, dims = 1L, drop = TRUE)

    list(
      add = a,
      sub = b,
      mul = c,
      div = d,
      neg = e,
      abs = ax,
      sqrt = sq,
      log = lg,
      exp = ex,
      expm1 = em1,
      log1p = l1p,
      logistic = lo,
      sine = si,
      cosine = co,
      tan = ta,
      tanh = th,
      floor = fl,
      ceil = ce,
      maximum = mx,
      minimum = mn,
      power = pw,
      eq = eq,
      ne = ne,
      gt = gt,
      ge = ge,
      lt = lt,
      le = le,
      and = aand,
      or = oor,
      xor = xxor,
      not = nn,
      select = sel,
      sum = sm,
      reduce_sum_1 = rs1,
      reduce_sum_2 = rs2,
      reduce_prod = rp,
      reduce_any = ra,
      reduce_all = rall,
      reduce_max = rmax,
      reduce_min = rmin
    )
  }

  templates <- list(
    x_f64 = nv_tensor(array(0, dim = c(2L, 3L)), dtype = "f64", shape = c(2L, 3L)),
    y_f64 = nv_tensor(array(0, dim = c(2L, 3L)), dtype = "f64", shape = c(2L, 3L)),
    x_i32 = nv_tensor(array(0L, dim = c(2L, 3L)), dtype = "i32", shape = c(2L, 3L)),
    y_i32 = nv_tensor(array(0L, dim = c(2L, 3L)), dtype = "i32", shape = c(2L, 3L)),
    p = nv_scalar(FALSE, dtype = "pred"),
    q = nv_scalar(FALSE, dtype = "pred")
  )

  run1 <- list(
    args = list(
      x_f64 = matrix(c(-0.2, -0.1, -0.3, -0.4, -0.5, -0.6), nrow = 2, byrow = TRUE),
      y_f64 = matrix(c(0.9, -0.8, 0.7, 0.6, -0.5, 0.4), nrow = 2, byrow = TRUE),
      x_i32 = matrix(c(1L, 2L, 3L, 4L, 5L, 6L), nrow = 2, byrow = TRUE),
      y_i32 = matrix(c(1L, 0L, 3L, 9L, 5L, 7L), nrow = 2, byrow = TRUE),
      p = TRUE,
      q = FALSE
    ),
    info = "run1"
  )

  run2 <- list(
    args = list(
      x_f64 = matrix(c(0.2, 0.1, 0.3, 0.4, 0.5, 0.6), nrow = 2, byrow = TRUE),
      y_f64 = matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), nrow = 2, byrow = TRUE),
      x_i32 = matrix(c(0L, 1L, 0L, 1L, 0L, 1L), nrow = 2, byrow = TRUE),
      y_i32 = matrix(c(1L, 1L, 0L, 0L, 1L, 1L), nrow = 2, byrow = TRUE),
      p = FALSE,
      q = FALSE
    ),
    info = "run2"
  )

  expect_quickr_matches_pjrt_fn(core_ops, templates, list(run1, run2), tolerance = 1e-6)
})

test_that("quickr pipeline matches PJRT: fill/iota/reverse/concatenate/convert/broadcast/transpose/reshape", {
  skip_if_no_quickr_or_pjrt()

  shape_ops <- function(s, v1, v2, x_i32, x21_i32, x13_i32, x3, x_pred) {
    f0 <- nv_fill(3.25, shape = integer(), dtype = "f64")
    f1 <- nv_fill(2L, shape = 4L, dtype = "i32")
    f2 <- nv_fill(TRUE, shape = c(2L, 3L), dtype = "pred")

    i1 <- nv_iota(dim = 1L, dtype = "i32", shape = 5L, start = 1L)
    i2 <- nv_iota(dim = 2L, dtype = "f64", shape = c(2L, 3L), start = 0L)
    i3_empty <- nv_iota(dim = 2L, dtype = "i32", shape = c(0L, 3L), start = 0L)
    i4_noloop <- nv_iota(dim = 2L, dtype = "i32", shape = c(1L, 4L, 1L), start = 0L)

    r1 <- nv_reverse(i1, dims = 1L)
    r2 <- nv_reverse(x_i32, dims = 2L)
    r3 <- nv_reverse(x3, dims = c(1L, 3L))

    c1 <- nv_concatenate(v1, v2, dimension = 1L)
    c2 <- nv_concatenate(x_i32, x_i32, dimension = 1L)
    c3 <- nv_concatenate(x3, x3, dimension = 2L)
    c4 <- nv_concatenate(x13_i32, x13_i32, dimension = 1L)

    s_i32 <- nv_convert(s, dtype = "i32")
    p_i32 <- nv_convert(x_pred, dtype = "i32")
    i32_pred <- nv_convert(x_i32, dtype = "pred")
    # Use nvl_* to force a convert node even when it's a no-op.
    i32_i32 <- nvl_convert(x_i32, dtype = "i32")
    pred_pred <- nvl_convert(x_pred, dtype = "pred")
    i32_f64 <- nvl_convert(x_i32, dtype = "f64")

    b0 <- nv_broadcast_to(s, shape = c(2L, 1L, 3L))
    b1 <- nvl_broadcast_in_dim(x_i32, shape = c(2L, 3L), broadcast_dimensions = c(1L, 2L))
    b2 <- nv_broadcast_to(x21_i32, shape = c(2L, 3L))
    b3 <- nvl_broadcast_in_dim(x_i32, shape = c(2L, 3L, 1L), broadcast_dimensions = c(1L, 2L))

    t0 <- nv_transpose(x_i32, permutation = c(2L, 1L))
    t1 <- nv_transpose(x_i32, permutation = c(1L, 2L))

    rs <- nv_reshape(s, shape = c(1L, 1L))
    r2a <- nv_reshape(x_i32, shape = c(3L, 2L))
    r3a <- nv_reshape(x3, shape = c(3L, 4L))

    list(
      fill0 = f0,
      fill1 = f1,
      fill2 = f2,
      iota1 = i1,
      iota2 = i2,
      iota_empty = i3_empty,
      iota_noloop = i4_noloop,
      reverse1 = r1,
      reverse2 = r2,
      reverse3 = r3,
      concatenate1 = c1,
      concatenate2 = c2,
      concatenate3 = c3,
      concatenate_1row = c4,
      convert_s = s_i32,
      convert_pred_i32 = p_i32,
      convert_i32_pred = i32_pred,
      convert_i32_i32 = i32_i32,
      convert_pred_pred = pred_pred,
      convert_i32_f64 = i32_f64,
      broadcast_scalar = b0,
      broadcast_id = b1,
      broadcast_2x1_to_2x3 = b2,
      broadcast_add_dim = b3,
      transpose = t0,
      transpose_id = t1,
      reshape_scalar_fast = rs,
      reshape_2d = r2a,
      reshape_3d = r3a
    )
  }

  templates <- list(
    s = nv_scalar(0.0, dtype = "f64"),
    v1 = nv_tensor(array(0L, dim = 3L), dtype = "i32", shape = 3L),
    v2 = nv_tensor(array(0L, dim = 2L), dtype = "i32", shape = 2L),
    x_i32 = nv_tensor(array(0L, dim = c(2L, 3L)), dtype = "i32", shape = c(2L, 3L)),
    x21_i32 = nv_tensor(array(0L, dim = c(2L, 1L)), dtype = "i32", shape = c(2L, 1L)),
    x13_i32 = nv_tensor(array(0L, dim = c(1L, 3L)), dtype = "i32", shape = c(1L, 3L)),
    x3 = nv_tensor(array(0L, dim = c(2L, 2L, 3L)), dtype = "i32", shape = c(2L, 2L, 3L)),
    x_pred = nv_tensor(array(FALSE, dim = c(2L, 3L)), dtype = "pred", shape = c(2L, 3L))
  )

  run <- list(
    args = list(
      s = 2.5,
      v1 = c(1L, 2L, 3L),
      v2 = c(4L, 5L),
      x_i32 = matrix(1:6, nrow = 2, byrow = TRUE),
      x21_i32 = matrix(c(1L, 2L), nrow = 2),
      x13_i32 = matrix(c(1L, 2L, 3L), nrow = 1L),
      x3 = array(1:12, dim = c(2L, 2L, 3L)),
      x_pred = matrix(c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE), nrow = 2, byrow = TRUE)
    ),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(shape_ops, templates, list(run))
})

test_that("quickr pipeline matches PJRT: broadcast + iota slice assignments compile at rank 3", {
  skip_if_no_quickr_or_pjrt()

  # Regression test: slice assignments at rank 3 with singleton dims (from
  # broadcast) should compile and match PJRT.
  b <- 4L
  n <- 7L
  m <- 5L

  fn <- function(x21) {
    bx <- nv_broadcast_to(x21, shape = c(b, n, m))
    ii <- nv_iota(dim = 2L, dtype = "i32", shape = c(b, n, m), start = 0L)
    bx + nv_convert(ii, dtype = "f64")
  }

  templates <- list(x21 = nv_tensor(array(0, dim = c(b, 1L, m)), dtype = "f64", shape = c(b, 1L, m)))

  run <- list(
    args = list(x21 = array(rnorm(b * m), dim = c(b, 1L, m))),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(fn, templates, list(run))
})

test_that("quickr pipeline matches PJRT: zero-length dims (reverse/concatenate/boolean reductions)", {
  skip_if_no_quickr_or_pjrt()

  # This exercises:
  # - 0-extent *inputs* (compile-time interface)
  # - dynamic_slice with 0 slice sizes (0-extent intermediates)
  # - reverse/concatenate/reduce_any/reduce_all on 0-extent arrays.
  zero_dim_ops <- function(X, empty_cols, empty_rows) {
    one <- nv_scalar(1L, dtype = "i32")
    ds_empty_cols <- nvl_dynamic_slice(X, one, one, slice_sizes = c(2L, 0L))
    ds_empty_rows <- nvl_dynamic_slice(X, one, one, slice_sizes = c(0L, 3L))

    p_cols <- nv_reverse(empty_cols, dims = 2L) > 0L

    list(
      ds_empty_cols = ds_empty_cols,
      ds_empty_rows = ds_empty_rows,
      cat_rows = nv_concatenate(empty_rows, X, dimension = 1L),
      cat_cols = nv_concatenate(X, empty_cols, dimension = 2L),
      any_cols = nv_reduce_any(p_cols, dims = 2L, drop = TRUE),
      all_cols = nv_reduce_all(p_cols, dims = 2L, drop = TRUE),
      any_rows = nv_reduce_any(empty_rows > 0L, dims = 1L, drop = TRUE),
      all_rows = nv_reduce_all(empty_rows > 0L, dims = 1L, drop = TRUE)
    )
  }

  X <- matrix(1:6, nrow = 2, byrow = TRUE)
  empty_cols <- matrix(integer(), nrow = 2L, ncol = 0L)
  empty_rows <- matrix(integer(), nrow = 0L, ncol = 3L)
  graph <- trace_fn(
    zero_dim_ops,
    list(
      X = nv_tensor(array(0L, dim = c(2L, 3L)), dtype = "i32", shape = c(2L, 3L)),
      empty_cols = nv_tensor(array(0L, dim = c(2L, 0L)), dtype = "i32", shape = c(2L, 0L)),
      empty_rows = nv_tensor(array(0L, dim = c(0L, 3L)), dtype = "i32", shape = c(0L, 3L))
    )
  )

  out <- expect_quickr_matches_pjrt(graph, X, empty_cols, empty_rows)
  pjrt <- out$out_pjrt

  expect_equal(pjrt$cat_rows, X)
  expect_equal(pjrt$cat_cols, X)
  expect_equal(as.logical(pjrt$any_cols), rep.int(FALSE, 2L))
  expect_equal(as.logical(pjrt$all_cols), rep.int(TRUE, 2L))
  expect_equal(as.logical(pjrt$any_rows), rep.int(FALSE, 3L))
  expect_equal(as.logical(pjrt$all_rows), rep.int(TRUE, 3L))
})

test_that("quickr pipeline matches PJRT: reduction branch coverage", {
  skip_if_no_quickr_or_pjrt()

  reduce_ops <- function(v_i32, v_pred, x_i32, x_pred, x3_i32, x3_pred, v_f64, x_f64, x3_f64) {
    sum_noop <- nv_reduce_sum(v_i32, dims = integer(), drop = FALSE)
    sum_rank1_keepdim <- nv_reduce_sum(v_i32, dims = 1L, drop = FALSE)
    sum_rank2_full_drop <- nv_reduce_sum(x_i32, dims = c(1L, 2L), drop = TRUE)
    sum_rank2_full_keep <- nv_reduce_sum(x_i32, dims = c(1L, 2L), drop = FALSE)

    prod_i32 <- nv_reduce_prod(x_i32, dims = 2L, drop = TRUE)
    prod_rank3_full_keep <- nv_reduce_prod(x3_i32, dims = 1:3, drop = FALSE)

    any_rank1_keepdim <- nv_reduce_any(v_pred, dims = 1L, drop = FALSE)
    any_rank2_keepdim <- nv_reduce_any(x_pred, dims = 2L, drop = FALSE)
    all_rank2_drop <- nv_reduce_all(x_pred, dims = 1L, drop = TRUE)
    all_rank3_full_keepdim <- nv_reduce_all(x3_pred, dims = 1:3, drop = FALSE)

    max_rank1_drop <- nv_reduce_max(v_f64, dims = 1L, drop = TRUE)
    min_rank1_keepdim <- nv_reduce_min(v_f64, dims = 1L, drop = FALSE)
    max_rank2_keepdim <- nv_reduce_max(x_f64, dims = 2L, drop = FALSE)
    max_rank3_full_drop <- nv_reduce_max(x3_f64, dims = 1:3, drop = TRUE)
    min_rank3_full_keep <- nv_reduce_min(x3_f64, dims = 1:3, drop = FALSE)

    list(
      sum_noop = sum_noop,
      sum_rank1_keepdim = sum_rank1_keepdim,
      sum_rank2_full_drop = sum_rank2_full_drop,
      sum_rank2_full_keep = sum_rank2_full_keep,
      prod_i32 = prod_i32,
      prod_rank3_full_keep = prod_rank3_full_keep,
      any_rank1_keepdim = any_rank1_keepdim,
      any_rank2_keepdim = any_rank2_keepdim,
      all_rank2_drop = all_rank2_drop,
      all_rank3_full_keepdim = all_rank3_full_keepdim,
      max_rank1_drop = max_rank1_drop,
      min_rank1_keepdim = min_rank1_keepdim,
      max_rank2_keepdim = max_rank2_keepdim,
      max_rank3_full_drop = max_rank3_full_drop,
      min_rank3_full_keep = min_rank3_full_keep
    )
  }

  templates <- list(
    v_i32 = nv_tensor(array(0L, dim = 4L), dtype = "i32", shape = 4L),
    v_pred = nv_tensor(array(FALSE, dim = 4L), dtype = "pred", shape = 4L),
    x_i32 = nv_tensor(array(0L, dim = c(2L, 3L)), dtype = "i32", shape = c(2L, 3L)),
    x_pred = nv_tensor(array(FALSE, dim = c(2L, 3L)), dtype = "pred", shape = c(2L, 3L)),
    x3_i32 = nv_tensor(array(0L, dim = c(2L, 2L, 3L)), dtype = "i32", shape = c(2L, 2L, 3L)),
    x3_pred = nv_tensor(array(FALSE, dim = c(2L, 2L, 3L)), dtype = "pred", shape = c(2L, 2L, 3L)),
    v_f64 = nv_tensor(array(0, dim = 4L), dtype = "f64", shape = 4L),
    x_f64 = nv_tensor(array(0, dim = c(2L, 3L)), dtype = "f64", shape = c(2L, 3L)),
    x3_f64 = nv_tensor(array(0, dim = c(2L, 2L, 3L)), dtype = "f64", shape = c(2L, 2L, 3L))
  )

  run <- list(
    args = list(
      v_i32 = c(1L, 2L, 3L, 4L),
      v_pred = c(TRUE, FALSE, TRUE, TRUE),
      x_i32 = matrix(1:6, nrow = 2, byrow = TRUE),
      x_pred = matrix(c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE), nrow = 2, byrow = TRUE),
      x3_i32 = array(1:12, dim = c(2L, 2L, 3L)),
      x3_pred = array(rep.int(c(TRUE, FALSE), 6L), dim = c(2L, 2L, 3L)),
      v_f64 = c(-0.5, 0.25, -0.1, 0.75),
      x_f64 = matrix(c(-1, 2, -3, 4, -5, 6), nrow = 2, byrow = TRUE),
      x3_f64 = array(seq(-6, 5, by = 1), dim = c(2L, 2L, 3L))
    ),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(reduce_ops, templates, list(run))
})

test_that("quickr pipeline matches PJRT: indexing ops (slice/update/pad) + gather/scatter", {
  skip_if_no_quickr_or_pjrt()

  idx_ops <- function(x, X, X3, s, r, c, a, b, d, upd_s, upd, upd2, padv, idx_vec, sc_idx_vec) {
    ss0 <- nv_static_slice(s, start_indices = integer(), limit_indices = integer(), strides = integer())
    ss1 <- nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 2L)
    ss2 <- nv_static_slice(X, start_indices = c(1L, 2L), limit_indices = c(2L, 3L), strides = c(1L, 1L))
    ss3 <- nv_static_slice(
      X3,
      start_indices = c(2L, 1L, 1L),
      limit_indices = c(3L, 3L, 2L),
      strides = c(1L, 1L, 1L)
    )

    ds0 <- nvl_dynamic_slice(s, slice_sizes = integer())
    ds1 <- nvl_dynamic_slice(x, s, slice_sizes = 3L)
    ds2 <- nvl_dynamic_slice(X, r, c, slice_sizes = c(2L, 2L))
    ds3 <- nvl_dynamic_slice(X3, a, b, d, slice_sizes = c(2L, 2L, 1L))

    dus0 <- nvl_dynamic_update_slice(s, upd_s)
    dus1 <- nvl_dynamic_update_slice(x, upd, s)
    dus2 <- nvl_dynamic_update_slice(X, upd2, r, c)

    p0 <- nv_pad(s, padv, edge_padding_low = integer(), edge_padding_high = integer())
    p1 <- nv_pad(x, padv, edge_padding_low = 2L, edge_padding_high = 1L, interior_padding = 0L)
    p2 <- nv_pad(X, padv, edge_padding_low = c(1L, 2L), edge_padding_high = c(0L, 1L))
    p3 <- nv_pad(
      X3,
      padv,
      edge_padding_low = c(1L, 0L, 1L),
      edge_padding_high = c(0L, 1L, 0L),
      interior_padding = c(0L, 1L, 0L)
    )

    # Gather/scatter via user-facing subsetting APIs.
    g <- X3[idx_vec, , ]

    base <- nv_iota(dim = 1L, dtype = "i32", shape = 6L, start = 0L)
    scattered_overwrite <- nv_subset_assign(base, sc_idx_vec, value = upd)

    sc_idx <- nv_reshape(sc_idx_vec, c(2L, 1L))
    scattered_add <- nvl_scatter(
      base,
      sc_idx,
      upd,
      update_window_dims = integer(),
      inserted_window_dims = 1L,
      input_batching_dims = integer(),
      scatter_indices_batching_dims = integer(),
      scatter_dims_to_operand_dims = 1L,
      index_vector_dim = 2L,
      update_computation = function(old, new) old + new
    )

    list(
      static0 = ss0,
      static1 = ss1,
      static2 = ss2,
      static3 = ss3,
      dynamic0 = ds0,
      dynamic1 = ds1,
      dynamic2 = ds2,
      dynamic3 = ds3,
      update0 = dus0,
      update1 = dus1,
      update2 = dus2,
      pad0 = p0,
      pad1 = p1,
      pad2 = p2,
      pad3 = p3,
      gather = g,
      scatter_overwrite = scattered_overwrite,
      scatter_add = scattered_add
    )
  }

  templates <- list(
    x = nv_tensor(array(0L, dim = 6L), dtype = "i32", shape = 6L),
    X = nv_tensor(array(0L, dim = c(2L, 3L)), dtype = "i32", shape = c(2L, 3L)),
    X3 = nv_tensor(array(0L, dim = c(3L, 4L, 2L)), dtype = "i32", shape = c(3L, 4L, 2L)),
    s = nv_scalar(0L, dtype = "i32"),
    r = nv_scalar(0L, dtype = "i32"),
    c = nv_scalar(0L, dtype = "i32"),
    a = nv_scalar(0L, dtype = "i32"),
    b = nv_scalar(0L, dtype = "i32"),
    d = nv_scalar(0L, dtype = "i32"),
    upd_s = nv_scalar(0L, dtype = "i32"),
    upd = nv_tensor(array(0L, dim = 2L), dtype = "i32", shape = 2L),
    upd2 = nv_tensor(array(0L, dim = c(2L, 2L)), dtype = "i32", shape = c(2L, 2L)),
    padv = nv_scalar(0L, dtype = "i32"),
    idx_vec = nv_tensor(array(0L, dim = 2L), dtype = "i32", shape = 2L),
    sc_idx_vec = nv_tensor(array(0L, dim = 2L), dtype = "i32", shape = 2L)
  )

  X3 <- array(sample.int(100, 24, replace = TRUE), dim = c(3L, 4L, 2L))

  common_args <- list(
    x = 1:6,
    X = matrix(1:6, nrow = 2, byrow = TRUE),
    X3 = X3,
    r = 2L,
    c = 2L,
    a = 0L,
    b = 100L,
    d = 2L,
    upd_s = 11L,
    upd = c(10L, 20L),
    upd2 = matrix(c(1L, 2L, 3L, 4L), nrow = 2, byrow = TRUE),
    padv = 0L,
    idx_vec = c(0L, 4L),
    sc_idx_vec = c(2L, 6L)
  )

  run_clamp_low <- list(
    args = c(common_args, list(s = -100L)),
    info = "clamp_low"
  )

  run_clamp_high <- list(
    args = c(common_args, list(s = 100L)),
    info = "clamp_high"
  )

  expect_quickr_matches_pjrt_fn(idx_ops, templates, list(run_clamp_low, run_clamp_high))
})

test_that("quickr pipeline matches PJRT: gather can return a scalar", {
  skip_if_no_quickr_or_pjrt()

  gather_scalar <- function(x, idx) {
    nvl_gather(
      x,
      idx,
      slice_sizes = 1L,
      offset_dims = integer(),
      collapsed_slice_dims = 1L,
      operand_batching_dims = integer(),
      start_indices_batching_dims = integer(),
      start_index_map = 1L,
      index_vector_dim = 1L
    )
  }

  templates <- list(
    x = nv_tensor(array(0L, dim = 5L), dtype = "i32", shape = 5L),
    idx = nv_tensor(array(0L, dim = 1L), dtype = "i32", shape = 1L)
  )

  run_low <- list(args = list(x = 1:5, idx = 0L), info = "clamp_low")
  run_high <- list(args = list(x = 1:5, idx = 100L), info = "clamp_high")

  expect_quickr_matches_pjrt_fn(gather_scalar, templates, list(run_low, run_high))
})

test_that("quickr pipeline matches PJRT: dot_general variants", {
  skip_if_no_quickr_or_pjrt()

  dot_ops <- function(a, b, a1, b1, a_dot, b_dot, a_mv, b_mv, lhs, rhs, lhs_b, rhs_b) {
    s <- nvl_dot_general(
      a,
      b,
      contracting_dims = list(integer(), integer()),
      batching_dims = list(integer(), integer())
    )
    outer <- nvl_dot_general(
      a1,
      b1,
      contracting_dims = list(integer(), integer()),
      batching_dims = list(integer(), integer())
    )
    dot <- nvl_dot_general(a_dot, b_dot, contracting_dims = list(1L, 1L), batching_dims = list(integer(), integer()))
    mv <- nvl_dot_general(a_mv, b_mv, contracting_dims = list(2L, 1L), batching_dims = list(integer(), integer()))
    mm <- nvl_dot_general(lhs, rhs, contracting_dims = list(2L, 1L), batching_dims = list(integer(), integer()))
    batch <- nvl_dot_general(lhs_b, rhs_b, contracting_dims = list(3L, 2L), batching_dims = list(1L, 1L))

    list(scalar = s, outer = outer, dot = dot, mv = mv, mm = mm, batch = batch)
  }

  templates <- list(
    a = nv_scalar(0.0, dtype = "f64"),
    b = nv_scalar(0.0, dtype = "f64"),
    a1 = nv_tensor(array(0L, dim = 2L), dtype = "i32", shape = 2L),
    b1 = nv_tensor(array(0L, dim = 3L), dtype = "i32", shape = 3L),
    a_dot = nv_tensor(array(0L, dim = 3L), dtype = "i32", shape = 3L),
    b_dot = nv_tensor(array(0L, dim = 3L), dtype = "i32", shape = 3L),
    a_mv = nv_tensor(array(0L, dim = c(2L, 3L)), dtype = "i32", shape = c(2L, 3L)),
    b_mv = nv_tensor(array(0L, dim = 3L), dtype = "i32", shape = 3L),
    lhs = nv_tensor(array(0, dim = c(2L, 3L)), dtype = "f64", shape = c(2L, 3L)),
    rhs = nv_tensor(array(0, dim = c(3L, 4L)), dtype = "f64", shape = c(3L, 4L)),
    lhs_b = nv_tensor(array(0, dim = c(2L, 3L, 4L)), dtype = "f64", shape = c(2L, 3L, 4L)),
    rhs_b = nv_tensor(array(0, dim = c(2L, 4L, 5L)), dtype = "f64", shape = c(2L, 4L, 5L))
  )

  run <- list(
    args = list(
      a = 2,
      b = 3,
      a1 = 1:2,
      b1 = c(10L, 20L, 30L),
      a_dot = 1:3,
      b_dot = 4:6,
      a_mv = matrix(1:6, nrow = 2, byrow = TRUE),
      b_mv = 1:3,
      lhs = matrix(c(0.2, -0.1, 0.3, 0.4, 0.5, -0.2), nrow = 2, byrow = TRUE),
      rhs = matrix(c(0.1, 0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.9, -1.0, 1.1, -1.2), nrow = 3, byrow = TRUE),
      lhs_b = array(0.1 * (1:24), dim = c(2L, 3L, 4L)),
      rhs_b = array(0.05 * (1:40), dim = c(2L, 4L, 5L))
    ),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(dot_ops, templates, list(run))
})

test_that("quickr pipeline matches PJRT: control flow (if/while)", {
  skip_if_no_quickr_or_pjrt()

  cf_ops <- function(p) {
    if_s <- nv_if(p, 1L, 2L)
    if_a <- nv_if(
      p,
      nv_fill(1L, shape = c(2L, 2L), dtype = "i32"),
      nv_fill(2L, shape = c(2L, 2L), dtype = "i32")
    )

    w1 <- nv_while(
      init = list(i = 0L, total = 0L),
      cond = function(i, total) i < 5L,
      body = function(i, total) list(i = i + 1L, total = total + i)
    )

    w2 <- nv_while(
      init = list(i = 0L, v = nv_iota(dim = 1L, dtype = "i32", shape = 5L, start = 0L)),
      cond = function(i, v) i < 3L,
      body = function(i, v) list(i = i + 1L, v = v + 2L)
    )

    list(if_s = if_s, if_a = if_a, w1 = w1, w2 = w2)
  }

  templates <- list(p = nv_scalar(FALSE, dtype = "pred"))

  run_true <- list(args = list(p = TRUE), info = "p=TRUE")
  run_false <- list(args = list(p = FALSE), info = "p=FALSE")

  expect_quickr_matches_pjrt_fn(cf_ops, templates, list(run_true, run_false))
})
