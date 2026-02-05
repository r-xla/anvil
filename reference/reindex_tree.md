# Reindex Tree

Reassigns leaf indices so they form a contiguous sequence starting from
the current counter value. This is used internally after filtering nodes
from a tree (e.g. via
[`filter_list_node()`](https://r-xla.github.io/anvil/reference/filter_list_node.md))
to ensure leaf indices still map correctly to positions in a flat list.
Not intended for direct use.

## Usage

``` r
reindex_tree(x, counter)
```

## Arguments

- x:

  (`Node`)  
  A tree node to reindex.

- counter:

  (environment)  
  A mutable counter created by `new_counter()`.

## Value

A new `Node` with updated leaf indices.
