# Tree Path

Returns the human-readable path for a single leaf node identified by its
flat index. Only descends into the branch containing the target leaf,
making it efficient for error reporting.

## Usage

``` r
tree_path(node, i, prefix = "")
```

## Arguments

- node:

  (`Node`)  
  A tree node as returned by
  [`build_tree()`](https://r-xla.github.io/anvl/reference/build_tree.md).

- i:

  (`integer(1)`)  
  The flat index of the leaf (as stored in `LeafNode$i`).

- prefix:

  (`character(1)`)  
  Path prefix. Used internally during recursion; callers should leave as
  `""`.

## Value

A scalar `character` string.

## See also

[`build_tree()`](https://r-xla.github.io/anvl/reference/build_tree.md),
[`flatten()`](https://r-xla.github.io/anvl/reference/flatten.md)
