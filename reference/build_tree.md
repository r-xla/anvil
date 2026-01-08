# Build Tree

Build a tree structure from a nested object for tracking structure
during flattening/unflattening.

## Usage

``` r
build_tree(x, counter = NULL)
```

## Arguments

- x:

  Object to build tree from.

- counter:

  Internal counter for leaf indices.

## Value

A tree node (LeafNode or ListNode).
