# Scatter and Gather Backward Implementation Notes

This document captures the learnings and design decisions from implementing backward rules for `nvl_gather` and `nvl_scatter` in anvil.

## Overview

The gather and scatter operations are inverse operations:
- **Gather**: Reads values from specified positions in a source tensor
- **Scatter**: Writes values to specified positions in a target tensor

Their backward rules are also inverse to each other:
- **Gather backward** uses scatter (with addition) to accumulate gradients back to source positions
- **Scatter backward** uses gather to retrieve gradients from target positions

## Key Parameters

### `indices_are_sorted`
- When `TRUE`: Hints to XLA that indices are in sorted order, enabling optimizations
- Can only be reliably set if indices are static (known at trace time)
- For `nv_subset` with ranges (e.g., `x[2:5]`), this is always `TRUE`
- For `nv_subset` with list indices (e.g., `x[list(1, 3, 5)]`), we conservatively set `FALSE`

### `unique_indices`
- When `TRUE`: Guarantees each index appears at most once (no overlapping writes)
- **Critical for scatter backward**: Without unique indices, scatter behavior is non-deterministic
- For single-element writes (e.g., `x[2] <- val`), this is `TRUE`
- For scattered writes (e.g., `x[c(1, 3)] <- vals`), we conservatively set `FALSE`

### `fill_value` (nvl_gather)
- When specified: Out-of-bounds indices return `fill_value` instead of reading from operand
- When `NULL` (default): Behavior is implementation-defined (XLA may clamp or return zeros)
- **Enables well-defined backward pass**:
  - In-bounds positions: gradient flows back to operand via scatter_add
  - Out-of-bounds positions: gradient is 0 (output doesn't depend on operand, it's a constant)
- This is semantically correct because:
  - `y[i] = fill_value` (constant) → `dy/dx = 0`
  - `y[i] = x[idx]` → `dy/dx[idx] = dy[i]`

## Gather Backward Rule

The backward of gather is scatter with addition:

```r
p_gather[["backward"]] <- function(inputs, outputs, grads, ...) {
  # grad w.r.t. operand = scatter_add(zeros, indices, grad)
  nvl_scatter(
    input = zeros_like(operand),
    scatter_indices = start_indices,
    update = grad,
    ...
    update_computation = function(old, new) nvl_add(old, new)  # accumulate!
  )
}
```

**Why addition?** If the same source position is gathered multiple times, all those gathered values contribute to the loss, so gradients must accumulate at that position.

Example:
```r
x = [a, b, c]
y = gather(x, [2, 2])  # y = [b, b]
loss = sum(y)          # loss = 2*b

# Forward: y[1] = x[2] = b, y[2] = x[2] = b
# Backward: d(loss)/d(x[2]) = d(loss)/d(y[1]) + d(loss)/d(y[2]) = 1 + 1 = 2
# So grad_x = [0, 2, 0]
```

## Scatter Backward Rule

The backward of scatter has two components:

1. **Gradient for input**: Scatter zeros to overwritten positions
   ```r
   # Positions that were overwritten don't contribute to output
   nvl_scatter(grad, indices, zeros(update_shape), ..., update = new)
   ```

2. **Gradient for update**: Gather from gradient at update positions
   ```r
   # Get gradient from positions where updates were written
   nvl_gather(grad, indices, ...)
   ```

**Requirement: unique_indices must be TRUE**

When indices overlap (same position written multiple times), XLA doesn't guarantee which write "wins". This makes the gradient ill-defined because:
- We don't know which update value is actually in the output
- The "winning" update's gradient should be the output gradient at that position
- The "losing" updates' gradients should be zero

JAX handles this with a complex ID-based scheme:
1. Attach unique IDs to each update
2. Scatter the IDs to see which update "wins" at each position
3. Gather the scattered IDs back to determine the winner
4. Mask updates based on whether they won

We chose to throw an error for `unique_indices = FALSE` instead of implementing this complexity.

## JAX Reference: Scatter JVP (for non-unique indices)

JAX's approach for non-unique indices in the JVP rule:

```python
# a) attach positive ID to each update, scatter the IDs
update_ids = iota(...) + 1  # IDs: 1, 2, 3, ...
scattered_ids = scatter(zeros, indices, update_ids)

# b) inverse gather to see which update "won" at each position
gathered_ids = gather(scattered_ids, indices)

# c) mask: only the "winning" update contributes
# An update "wins" if its ID equals the gathered ID at its position
mask = (update_ids == gathered_ids)
masked_updates = select(mask, updates, zeros)
masked_g_updates = select(mask, g_updates, zeros)

# d) scatter-add the masked values
val_out = scatter_add(masked_operand, indices, masked_updates)
tangent_out = scatter_add(masked_g_operand, indices, masked_g_updates)
```

This is the JVP, not the VJP. For our purposes (VJP), we use the transpose rule which requires `unique_indices = TRUE`.

## fill_value Implementation Details

### Forward Pass (rules-stablehlo.R)

When `fill_value` is specified, the stablehlo rule for gather:

1. Performs the normal gather operation
2. Computes an in-bounds mask for each index coordinate:
   - For each dimension `d` in `start_index_map`:
   - Valid range (0-based): `0 <= idx <= operand_shape[d] - slice_sizes[d]`
3. Creates a fill tensor with `fill_value` broadcast to result shape
4. Uses `hlo_select(in_bounds_mask, gather_result, fill_tensor)`

Helper functions:
- `.stablehlo_gather_apply_fill_value()`: Main logic for bounds checking and masking
- `.stablehlo_extract_index_coord()`: Extracts single coordinate from indices tensor
- `.stablehlo_broadcast_mask_to_result()`: Broadcasts mask to output shape

### Backward Pass (rules-backward.R)

When `fill_value` is specified, the backward rule:

1. Computes the same in-bounds mask (using 1-based indexing):
   - Valid range: `1 <= idx <= operand_shape[d] - slice_sizes[d] + 1`
2. Masks gradient to zero for out-of-bounds positions:
   - `grad_to_scatter = select(in_bounds_mask, grad, zeros)`
3. Clamps indices to valid range before scattering:
   - This ensures scatter doesn't ignore out-of-bounds indices
   - The clamped positions receive zero gradient anyway (from step 2)
4. Scatter-adds the masked gradient to operand

Helper functions:
- `.gather_compute_in_bounds_mask()`: Computes boolean mask for valid indices
- `.extract_index_coordinate()`: Extracts coordinate from indices tensor
- `.broadcast_mask_to_output()`: Broadcasts mask to output shape
- `.gather_clamp_indices()`: Clamps indices to valid range

### Why Clamp Indices in Backward?

XLA's scatter may ignore out-of-bounds indices entirely. By clamping:
- All indices point to valid positions (scatter processes them)
- The gradient at those positions is already zeroed (from masking)
- Result: correct gradient flow for in-bounds, zero for out-of-bounds

## Implementation Decisions

1. **Conservative flag settings**: When indices might be dynamic tensors, we set `indices_are_sorted = FALSE` and `unique_indices = FALSE` to be safe.

2. **Scatter backward requires unique_indices**: Rather than implementing the complex ID-based scheme from JAX, we throw an informative error for non-unique indices.

3. **1-based indexing**: Anvil uses 1-based indexing throughout. The stablehlo rules convert to 0-based when generating HLO.

## Testing Strategy

Tests verify:
1. **Gather backward**: Single element, range, list indices, and overlapping indices (gradient accumulation)
2. **Scatter backward**: Single element and range writes (input and update gradients separately)
3. **Error case**: Non-unique scattered indices throw an error during backward
4. **fill_value forward**: Out-of-bounds indices return fill_value (both positive and negative out-of-bounds)
5. **fill_value backward**: Gradient is 0 for out-of-bounds positions, gradient accumulation works correctly for duplicate in-bounds indices

## Future Work

1. **Non-unique scatter backward**: Could implement JAX's ID-based scheme if needed
2. **Static analysis for unique_indices**: Could detect unique list indices at trace time (e.g., `c(1, 3, 5)` is unique)
