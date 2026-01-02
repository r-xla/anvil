describe("inline_literals", {
  # Helper function to check inline_literals behavior
  check_inline_literals <- function(
    graph_fun,
    args,
    expected_constants_before = NULL,
    expected_constants_after = NULL,
    check_literals = NULL,
    check_structure = TRUE
  ) {
    graph <- do.call(trace_fn, c(list(f = graph_fun), args))

    if (!is.null(expected_constants_before)) {
      expect_length(graph@constants, expected_constants_before)
    }

    new_graph <- inline_literals(graph)

    if (!is.null(expected_constants_after)) {
      expect_length(new_graph@constants, expected_constants_after)
    }

    if (check_structure) {
      expect_equal(length(new_graph@calls), length(graph@calls))
      expect_equal(length(new_graph@inputs), length(graph@inputs))
      expect_equal(length(new_graph@outputs), length(graph@outputs))
      expect_identical(new_graph@in_tree, graph@in_tree)
      expect_identical(new_graph@out_tree, graph@out_tree)
    }

    if (!is.null(check_literals)) {
      check_literals(new_graph, graph)
    }

    return(list(original = graph, processed = new_graph))
  }

  it("converts scalar constants to GraphLiterals", {
    const_scalar <- nv_scalar(5)
    f <- function(x) {
      x + const_scalar
    }

    result <- check_inline_literals(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
      expected_constants_before = 1L,
      expected_constants_after = 0L,
      check_literals = function(new_graph, original_graph) {
        expect_true(is_graph_literal(new_graph@calls[[1L]]@inputs[[2L]]))
        expect_equal(new_graph@calls[[1L]]@inputs[[2L]]@aval@data, 5)
      }
    )
  })

  it("does not convert non-scalar constants", {
    const_tensor <- nv_tensor(c(1, 2, 3))
    f <- function(x) {
      x + const_tensor
    }

    result <- check_inline_literals(
      graph_fun = f,
      args = list(list(x = nv_tensor(c(4, 5, 6)))),
      expected_constants_before = 1L,
      expected_constants_after = 1L,
      check_literals = function(new_graph, original_graph) {
        expect_equal(length(shape(original_graph@constants[[1L]])), 1L)
        expect_true(is_graph_value(new_graph@constants[[1L]]))
        expect_false(is_graph_literal(new_graph@constants[[1L]]))
      }
    )
  })

  it("replaces all references to converted constants", {
    const_scalar <- nv_scalar(3)
    f <- function(x) {
      y <- x + const_scalar
      y * const_scalar
    }

    result <- check_inline_literals(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
      check_literals = function(new_graph, original_graph) {
        # All references to the constant should be GraphLiterals
        expect_true(is_graph_literal(new_graph@calls[[1L]]@inputs[[2L]]))
        expect_true(is_graph_literal(new_graph@calls[[2L]]@inputs[[2L]]))
        expect_equal(new_graph@calls[[1L]]@inputs[[2L]]@aval@data, 3)
        expect_equal(new_graph@calls[[2L]]@inputs[[2L]]@aval@data, 3)
      }
    )
  })

  it("processes subgraphs in higher-order primitives", {
    const_true <- nv_scalar(10)
    const_false <- nv_scalar(20)

    f <- function(x) {
      nv_if(x, const_true, const_false)
    }

    graph <- trace_fn(f, list(x = nv_scalar(TRUE)))
    expect_true(is_higher_order_primitive(graph@calls[[1L]]@primitive))
    true_graph <- graph@calls[[1L]]@params[["true_graph"]]
    false_graph <- graph@calls[[1L]]@params[["false_graph"]]
    expect_gte(length(true_graph@constants), 1L)
    expect_gte(length(false_graph@constants), 1L)

    new_graph <- inline_literals(graph)

    new_true_graph <- new_graph@calls[[1L]]@params[["true_graph"]]
    new_false_graph <- new_graph@calls[[1L]]@params[["false_graph"]]

    expect_length(new_true_graph@constants, 0L)
    expect_length(new_false_graph@constants, 0L)
    expect_true(is_graph_literal(new_true_graph@outputs[[1L]]))
    expect_true(is_graph_literal(new_false_graph@outputs[[1L]]))
    expect_equal(new_true_graph@outputs[[1L]]@aval@data, 10)
    expect_equal(new_false_graph@outputs[[1L]]@aval@data, 20)
  })

  it("processes subgraphs in while loops", {
    const_limit <- nv_scalar(5)

    f <- function() {
      nv_while(
        list(i = nv_scalar(0)),
        function(i) i < const_limit,
        function(i) list(i = i + nv_scalar(1))
      )
    }

    graph <- trace_fn(f, list())
    expect_true(is_higher_order_primitive(graph@calls[[1L]]@primitive))
    cond_graph <- graph@calls[[1L]]@params[["cond_graph"]]
    expect_length(cond_graph@constants, 1L)

    new_graph <- inline_literals(graph)

    new_cond_graph <- new_graph@calls[[1L]]@params[["cond_graph"]]
    expect_length(new_cond_graph@constants, 0L)

    if (length(new_cond_graph@calls) > 0L) {
      for (call in new_cond_graph@calls) {
        for (input in call@inputs) {
          if (is_graph_literal(input)) {
            expect_equal(input@aval@data, 5)
          }
        }
      }
    }
  })

  it("preserves graph structure", {
    const_scalar <- nv_scalar(2)
    f <- function(x) {
      y <- x * const_scalar
      y + const_scalar
    }

    check_inline_literals(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
      check_structure = TRUE
    )
  })

  it("handles graphs with no scalar constants", {
    f <- function(x, y) {
      x + y
    }

    check_inline_literals(
      graph_fun = f,
      args = list(list(x = nv_scalar(1), y = nv_scalar(2))),
      expected_constants_before = 0L,
      expected_constants_after = 0L,
      check_structure = TRUE
    )
  })

  it("handles multiple scalar constants", {
    const1 <- nv_scalar(1)
    const2 <- nv_scalar(2)
    const3 <- nv_scalar(3)

    f <- function(x) {
      x + const1 + const2 + const3
    }

    result <- check_inline_literals(
      graph_fun = f,
      args = list(list(x = nv_scalar(0))),
      expected_constants_before = 3L,
      expected_constants_after = 0L,
      check_literals = function(new_graph, original_graph) {
        expect_true(is_graph_literal(new_graph@calls[[1L]]@inputs[[2L]]))
        expect_equal(new_graph@calls[[1L]]@inputs[[2L]]@aval@data, 1)
      }
    )
  })

  it("preserves dtype and ambiguity of converted literals", {
    # Test with integer (ambiguous)
    const_int <- nv_scalar(5L)
    f <- function(x) {
      x + const_int
    }

    result <- check_inline_literals(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
      check_literals = function(new_graph, original_graph) {
        call <- new_graph@calls[[1L]]
        literal <- NULL
        for (input in call@inputs) {
          if (is_graph_literal(input)) {
            literal <- input
            break
          }
        }
        expect_false(is.null(literal))
        expect_equal(literal@aval@data, 5L)
        expect_true(literal@aval@ambiguous)
      }
    )

    # Test with logical (not ambiguous)
    const_logical <- nv_scalar(TRUE)
    f2 <- function(x) {
      x + const_logical
    }

    result2 <- check_inline_literals(
      graph_fun = f2,
      args = list(list(x = nv_scalar(1))),
      check_literals = function(new_graph, original_graph) {
        call <- new_graph@calls[[1L]]
        literal <- NULL
        for (input in call@inputs) {
          if (is_graph_literal(input)) {
            literal <- input
            break
          }
        }
        expect_false(is.null(literal))
        expect_equal(literal@aval@data, TRUE)
        expect_false(literal@aval@ambiguous)
      }
    )
  })

  it("processes nested subgraphs", {
    # Create a graph with nested nv_if calls
    const_inner_true <- nv_scalar(100)
    const_inner_false <- nv_scalar(200)
    const_outer_true <- nv_scalar(10)
    const_outer_false <- nv_scalar(20)

    f <- function(x, y) {
      nv_if(
        x,
        nv_if(y, const_inner_true, const_inner_false),
        nv_if(y, const_outer_true, const_outer_false)
      )
    }
    graph <- trace_fn(f, list(x = nv_scalar(TRUE), y = nv_scalar(TRUE)))

    # Verify original graph has nested subgraphs with constants
    expect_true(is_higher_order_primitive(graph@calls[[1L]]@primitive))
    outer_true_graph <- graph@calls[[1L]]@params[["true_graph"]]
    outer_false_graph <- graph@calls[[1L]]@params[["false_graph"]]

    # Both outer subgraphs should contain inner nv_if calls
    expect_true(is_higher_order_primitive(outer_true_graph@calls[[1L]]@primitive))
    expect_true(is_higher_order_primitive(outer_false_graph@calls[[1L]]@primitive))

    # Check that inner subgraphs have constants
    inner_true_graph_true <- outer_true_graph@calls[[1L]]@params[["true_graph"]]
    inner_true_graph_false <- outer_true_graph@calls[[1L]]@params[["false_graph"]]
    expect_gte(length(inner_true_graph_true@constants), 1L)
    expect_gte(length(inner_true_graph_false@constants), 1L)

    # Apply inline_literals
    new_graph <- inline_literals(graph)

    # Outer subgraphs should have their constants converted
    new_outer_true_graph <- new_graph@calls[[1L]]@params[["true_graph"]]
    new_outer_false_graph <- new_graph@calls[[1L]]@params[["false_graph"]]

    # Inner subgraphs should also have their constants converted
    expect_true(is_higher_order_primitive(new_outer_true_graph@calls[[1L]]@primitive))
    new_inner_true_graph_true <- new_outer_true_graph@calls[[1L]]@params[["true_graph"]]
    new_inner_true_graph_false <- new_outer_true_graph@calls[[1L]]@params[["false_graph"]]

    # All nested constants should be converted
    expect_length(new_inner_true_graph_true@constants, 0L)
    expect_length(new_inner_true_graph_false@constants, 0L)

    # The outputs of the innermost subgraphs should be GraphLiterals
    expect_true(is_graph_literal(new_inner_true_graph_true@outputs[[1L]]))
    expect_true(is_graph_literal(new_inner_true_graph_false@outputs[[1L]]))
    expect_equal(new_inner_true_graph_true@outputs[[1L]]@aval@data, 100)
    expect_equal(new_inner_true_graph_false@outputs[[1L]]@aval@data, 200)

    # Check the other branch as well
    expect_true(is_higher_order_primitive(new_outer_false_graph@calls[[1L]]@primitive))
    new_inner_false_graph_true <- new_outer_false_graph@calls[[1L]]@params[["true_graph"]]
    new_inner_false_graph_false <- new_outer_false_graph@calls[[1L]]@params[["false_graph"]]

    expect_length(new_inner_false_graph_true@constants, 0L)
    expect_length(new_inner_false_graph_false@constants, 0L)
    expect_true(is_graph_literal(new_inner_false_graph_true@outputs[[1L]]))
    expect_true(is_graph_literal(new_inner_false_graph_false@outputs[[1L]]))
    expect_equal(new_inner_false_graph_true@outputs[[1L]]@aval@data, 10)
    expect_equal(new_inner_false_graph_false@outputs[[1L]]@aval@data, 20)
  })
})

