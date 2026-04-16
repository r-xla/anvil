# tree_path

    Code
      tree_path(build_tree(list(x = 1)), 1L)
    Output
      [1] "x"
    Code
      tree_path(build_tree(list(l = list(a = 1, b = 2))), 1L)
    Output
      [1] "l$a"
    Code
      tree_path(build_tree(list(l = list(a = 1, b = 2))), 2L)
    Output
      [1] "l$b"
    Code
      tree_path(build_tree(list(l = list(1, 2))), 2L)
    Output
      [1] "l[[2]]"
    Code
      tree_path(build_tree(list(l = list(1, b = 2))), 1L)
    Output
      [1] "l[[1]]"
    Code
      tree_path(build_tree(list(l = list(1, b = 2))), 2L)
    Output
      [1] "l$b"
    Code
      tree_path(build_tree(list(l = list(list(a = 1)))), 1L)
    Output
      [1] "l[[1]]$a"
    Code
      tree_path(build_tree(list(x = 1, y = 2)), 2L)
    Output
      [1] "y"
    Code
      tree_path(build_tree(list(pair = list(list(a = 1), list(b = 2)))), 2L)
    Output
      [1] "pair[[2]]$b"

