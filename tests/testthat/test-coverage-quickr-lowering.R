test_that("quickr lowering helpers cover edge cases", {
  bad_node <- structure(list(), class = "FooNode")
  testthat::expect_error(quickr_tree_leaf_paths(bad_node), "Unsupported node type", fixed = FALSE)

  tree1 <- ListNode(list(LeafNode(1L), LeafNode(2L)), names = NULL)
  tree2 <- ListNode(list(LeafNode(1L), LeafNode(2L)), names = list(NULL, "a"))
  testthat::expect_length(quickr_tree_leaf_paths(tree1), 2L)
  testthat::expect_length(quickr_tree_leaf_paths(tree2), 2L)

  nms <- quickr_paths_to_names(list(list(), list(1L, "a")))
  testthat::expect_identical(nms[[1L]], "x")
  testthat::expect_true(nzchar(nms[[2L]]))

  testthat::expect_identical(quickr_dtype_to_r_ctor("i32"), "integer")
  testthat::expect_identical(quickr_dtype_to_r_ctor("i1"), "logical")
  testthat::expect_error(quickr_dtype_to_r_ctor("bad_dtype"), "Unsupported dtype", fixed = FALSE)

  testthat::expect_identical(quickr_scalar_cast(1.5, "f64"), 1.5)
  testthat::expect_identical(quickr_scalar_cast(2L, "i32"), 2L)
  testthat::expect_identical(quickr_scalar_cast(TRUE, "pred"), TRUE)
  testthat::expect_error(quickr_scalar_cast(1, "bad_dtype"), "Unsupported dtype", fixed = FALSE)

  testthat::expect_identical(quickr_zero_literal_for(AbstractTensor("i1", integer())), FALSE)
  testthat::expect_identical(quickr_zero_literal_for(AbstractTensor("f64", integer())), 0.0)
  testthat::expect_identical(quickr_zero_literal_for(AbstractTensor("i32", integer())), 0L)

  bad_rank <- AbstractTensor("f64", rep.int(1L, 6L))
  testthat::expect_error(quickr_aval_type_call(bad_rank), "rank 5", fixed = FALSE)
  testthat::expect_error(
    quickr_emit_full_like(as.name("out"), 0.0, rep.int(1L, 6L), AbstractTensor("f64", rep.int(1L, 6L))),
    "rank 5",
    fixed = FALSE
  )

  lit <- GraphLiteral(LiteralTensor(TRUE, shape = integer(), dtype = "i1", ambiguous = FALSE))
  testthat::expect_equal(quickr_expr_of_node(lit, hashtab()), TRUE)

  gval <- GraphValue(AbstractTensor("f64", integer()))
  testthat::expect_error(quickr_expr_of_node(gval, hashtab()), "not mapped", fixed = FALSE)

  out_sym <- as.name("out")
  lhs <- as.name("lhs")
  rhs <- as.name("rhs")

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      rep.int(1L, 6L),
      integer(),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(integer(), integer()),
      batching_dims = list(integer(), integer())
    ),
    "rank 5",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 3L),
      c(2L, 3L),
      c(2L, 3L),
      AbstractTensor("f64", c(2L, 3L)),
      contracting_dims = list(c(1L), c(1L, 2L)),
      batching_dims = list(integer(), integer())
    ),
    "contracting_dims",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 3L),
      c(2L, 3L),
      c(2L, 3L),
      AbstractTensor("f64", c(2L, 3L)),
      contracting_dims = list(integer(), integer()),
      batching_dims = list(c(1L), c(1L, 2L))
    ),
    "batching_dims",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L),
      c(2L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(NA_integer_), c(1L)),
      batching_dims = list(integer(), integer())
    ),
    "missing values",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L),
      c(2L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(2L), c(1L)),
      batching_dims = list(integer(), integer())
    ),
    "out of range",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 2L),
      c(2L, 2L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(1L, 1L), c(1L, 2L)),
      batching_dims = list(integer(), integer())
    ),
    "must be unique",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 2L),
      c(2L, 2L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(1L), c(1L)),
      batching_dims = list(c(1L), c(2L))
    ),
    "disjoint",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 2L),
      c(2L, 2L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(1L), c(1L)),
      batching_dims = list(c(2L), c(1L))
    ),
    "disjoint",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 2L),
      c(3L, 2L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(2L), c(2L)),
      batching_dims = list(c(1L), c(1L))
    ),
    "batch dim sizes differ",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 2L),
      c(3L, 3L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(2L), c(1L)),
      batching_dims = list(integer(), integer())
    ),
    "contracting dim sizes differ",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(2L, 3L),
      c(3L, 4L),
      c(1L, 1L),
      AbstractTensor("f64", c(1L, 1L)),
      contracting_dims = list(c(2L), c(1L)),
      batching_dims = list(integer(), integer())
    ),
    "output shape mismatch",
    fixed = FALSE
  )

  testthat::expect_type(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      integer(),
      integer(),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(integer(), integer()),
      batching_dims = list(integer(), integer())
    ),
    "list"
  )

  testthat::expect_type(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      integer(),
      c(3L),
      c(3L),
      AbstractTensor("f64", c(3L)),
      contracting_dims = list(integer(), integer()),
      batching_dims = list(integer(), integer())
    ),
    "list"
  )

  testthat::expect_type(
    quickr_emit_dot_general(
      out_sym,
      lhs,
      rhs,
      c(3L),
      c(3L),
      integer(),
      AbstractTensor("f64", integer()),
      contracting_dims = list(c(1L), c(1L)),
      batching_dims = list(integer(), integer())
    ),
    "list"
  )

  transpose_bad_rank <- try(
    quickr_emit_transpose(out_sym, lhs, c(1L), c(2L), AbstractTensor("f64", c(2L))),
    silent = TRUE
  )
  testthat::expect_true(inherits(transpose_bad_rank, "try-error"))

  transpose_bad_perm <- try(
    quickr_emit_transpose(out_sym, lhs, c(1L, 1L), c(2L, 2L), AbstractTensor("f64", c(2L, 2L))),
    silent = TRUE
  )
  testthat::expect_true(inherits(transpose_bad_perm, "try-error"))

  testthat::expect_type(
    quickr_emit_broadcast_in_dim(
      out_sym,
      lhs,
      c(2L, 3L),
      c(2L, 3L),
      c(1L, 2L),
      AbstractTensor("f64", c(2L, 3L))
    ),
    "list"
  )

  testthat::expect_error(
    quickr_emit_broadcast_in_dim(out_sym, lhs, c(2L), integer(), 1L, AbstractTensor("f64", integer())),
    "cannot broadcast to a scalar",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_broadcast_in_dim(
      out_sym,
      lhs,
      c(2L, 3L, 4L, 5L, 6L, 7L),
      rep.int(1L, 6L),
      1:6,
      AbstractTensor("f64", rep.int(1L, 6L))
    ),
    "rank 5",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_broadcast_in_dim(out_sym, lhs, c(2L, 3L), c(2L, 3L, 1L), 1L, AbstractTensor("f64", c(2L, 3L, 1L))),
    "must have length",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_broadcast_in_dim(
      out_sym,
      lhs,
      c(2L, 3L),
      c(2L, 3L, 4L),
      c(0L, 2L),
      AbstractTensor("f64", c(2L, 3L, 4L))
    ),
    "out of range",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_broadcast_in_dim(
      out_sym,
      lhs,
      c(2L, 3L),
      c(2L, 3L, 4L),
      c(1L, 1L),
      AbstractTensor("f64", c(2L, 3L, 4L))
    ),
    "must be unique",
    fixed = FALSE
  )

  testthat::expect_error(
    quickr_emit_broadcast_in_dim(
      out_sym,
      lhs,
      c(2L, 3L),
      c(4L, 3L),
      c(1L, 2L),
      AbstractTensor("f64", c(4L, 3L))
    ),
    "shape mismatch",
    fixed = FALSE
  )

  alloc <- quickr_emit_assign(out_sym, rlang::call2("double", 2L))
  update <- function(ii, jj, acc) rlang::call2("<-", acc, rlang::call2("+", acc, ii))
  testthat::expect_type(quickr_emit_reduce2_axis_loop(out_sym, 2L, 3L, 2L, TRUE, alloc, 0.0, 1L, update), "list")
  testthat::expect_type(
    quickr_emit_reduce2_axis_loop(out_sym, 2L, 3L, 2L, FALSE, alloc, 0.0, 1L, update),
    "list"
  )
  testthat::expect_type(quickr_emit_reduce2_axis_loop(out_sym, 2L, 3L, 1L, TRUE, alloc, 0.0, 1L, update), "list")

  testthat::expect_error(
    quickr_emit_reduce_sum(out_sym, lhs, integer(), 1L, TRUE, AbstractTensor("f64", integer())),
    "scalar reduction dims",
    fixed = FALSE
  )
  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, integer(), integer(), TRUE, AbstractTensor("f64", integer())),
    "list"
  )

  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(3L), integer(), TRUE, AbstractTensor("f64", c(3L))),
    "list"
  )
  testthat::expect_error(
    quickr_emit_reduce_sum(out_sym, lhs, c(3L), 2L, TRUE, AbstractTensor("f64", integer())),
    "rank-1",
    fixed = FALSE
  )

  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 3L), c(1L, 2L), FALSE, AbstractTensor("f64", c(1L, 1L))),
    "list"
  )
  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 3L), 2L, FALSE, AbstractTensor("f64", c(2L, 1L))),
    "list"
  )
  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 3L), 2L, TRUE, AbstractTensor("f64", c(2L))),
    "list"
  )
  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 3L), 1L, TRUE, AbstractTensor("f64", c(3L))),
    "list"
  )
  testthat::expect_error(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 3L), 3L, TRUE, AbstractTensor("f64", integer())),
    "rank-2",
    fixed = FALSE
  )

  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 2L, 2L), integer(), TRUE, AbstractTensor("f64", c(2L, 2L, 2L))),
    "list"
  )
  testthat::expect_error(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 2L, 2L), 1L, TRUE, AbstractTensor("f64", integer())),
    "full reductions",
    fixed = FALSE
  )
  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 2L, 2L), 1:3, TRUE, AbstractTensor("f64", integer())),
    "list"
  )
  testthat::expect_type(
    quickr_emit_reduce_sum(out_sym, lhs, c(2L, 2L, 2L), 1:3, FALSE, AbstractTensor("f64", rep.int(1L, 3L))),
    "list"
  )

  testthat::expect_error(
    quickr_emit_reshape(out_sym, lhs, rep.int(1L, 6L), rep.int(1L, 6L), AbstractTensor("f64", rep.int(1L, 6L))),
    "rank 5",
    fixed = FALSE
  )
  testthat::expect_error(
    quickr_emit_reshape(out_sym, lhs, c(2L, 3L), c(2L, 2L), AbstractTensor("f64", c(2L, 2L))),
    "sizes differ",
    fixed = FALSE
  )

  reg <- new.env(parent = emptyenv())
  quickr_register_prim_lowerer(reg, c("a", "b"), identity)
  testthat::expect_true(is.function(reg$a))
  testthat::expect_true(is.function(reg$b))
})

test_that("graph_to_quickr_r_function covers validation branches", {
  testthat::expect_error(graph_to_quickr_r_function(list()), "Graph", fixed = FALSE)

  g0 <- Graph(calls = list(), in_tree = NULL, out_tree = NULL, inputs = list(), outputs = list(), constants = list())
  testthat::expect_error(graph_to_quickr_r_function(g0), "non-NULL", fixed = FALSE)

  g_in1 <- GraphValue(AbstractTensor("f64", integer()))
  g_in2 <- GraphValue(AbstractTensor("f64", integer()))
  g_out <- GraphValue(AbstractTensor("f64", integer()))
  g_bad_tree <- Graph(
    calls = list(),
    in_tree = LeafNode(1L),
    out_tree = LeafNode(1L),
    inputs = list(g_in1, g_in2),
    outputs = list(g_out),
    constants = list()
  )
  testthat::expect_error(graph_to_quickr_r_function(g_bad_tree), "input tree size", fixed = FALSE)

  graph <- trace_fn(
    function(anvil_quickr_x) {
      anvil_quickr_x
    },
    list(anvil_quickr_x = nv_scalar(1.0, dtype = "f64"))
  )
  f <- graph_to_quickr_r_function(graph)
  testthat::expect_type(f, "closure")

  graph2 <- trace_fn(
    function(x) {
      nv_exp(x)
    },
    list(x = nv_scalar(1.0, dtype = "f64"))
  )
  testthat::expect_error(graph_to_quickr_r_function(graph2), "does not support", fixed = FALSE)

  graph3 <- trace_fn(
    function(x) {
      list(a = x, b = x + x)
    },
    list(x = nv_scalar(1.0, dtype = "f64"))
  )
  testthat::expect_error(graph_to_quickr_r_function(graph3, pack_output = FALSE), "pack_output", fixed = FALSE)

  graph4 <- trace_fn(
    function(x) {
      x
    },
    list(x = nv_scalar(1.0, dtype = "f64"))
  )
  graph4@constants <- list(GraphValue(AbstractTensor("f64", integer())))
  testthat::expect_error(graph_to_quickr_r_function(graph4), "concrete", fixed = FALSE)

  in_val <- GraphValue(AbstractTensor("f64", integer()))
  lit <- GraphLiteral(LiteralTensor(1.0, shape = integer(), dtype = "f64", ambiguous = FALSE))
  bad_call <- PrimitiveCall(primitive = p_add, inputs = list(in_val, in_val), params = list(), outputs = list(lit))
  graph5 <- Graph(
    calls = list(bad_call),
    in_tree = LeafNode(1L),
    out_tree = LeafNode(1L),
    inputs = list(in_val),
    outputs = list(lit),
    constants = list()
  )
  testthat::expect_error(graph_to_quickr_r_function(graph5), "non-GraphValue primitive outputs", fixed = FALSE)

  graph6 <- trace_fn(function(x) x, list(x = nv_scalar(1.0, dtype = "f64")))
  testthat::local_mocked_bindings(
    quickr_base_has_declare = function() FALSE,
    .env = environment(quickr_base_has_declare)
  )
  f2 <- graph_to_quickr_r_function(graph6, include_declare = TRUE)
  testthat::expect_true(is.function(environment(f2)$declare))
})

test_that("assert_quickr_installed_with errors when installed is FALSE", {
  testthat::expect_error(assert_quickr_installed_with(NULL, FALSE), "must be installed", fixed = FALSE)
})

test_that("graph_to_quickr_function errors on non-Graph", {
  testthat::expect_error(graph_to_quickr_function(1), "Graph", fixed = FALSE)
})

test_that("graph_to_quickr_function decodes logical and integer leaves", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) {
      list(flag = TRUE, out = x)
    },
    list(x = nv_scalar(1L, dtype = "i32"))
  )

  f <- graph_to_quickr_function(graph)
  out <- f(7L)
  testthat::expect_identical(out$flag, TRUE)
  testthat::expect_identical(out$out, 7L)
})
