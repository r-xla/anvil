# Primitives Reference

The table below shows all 68 primitives and which rules they implement.
Note that a rule might be missing because it was just not added yet, or
because it does not apply for a specific primitive.

| Name                   | StableHLO | Backward |
|:-----------------------|:----------|:--------:|
| abs                    | ✓         |    ✓     |
| add                    | ✓         |    ✓     |
| and                    | ✓         |    ✗     |
| atan2                  | ✓         |    ✗     |
| bitcast_convert        | ✓         |    ✗     |
| broadcast_in_dim       | ✓         |    ✓     |
| cbrt                   | ✓         |    ✓     |
| ceil                   | ✓         |    ✓     |
| clamp                  | ✓         |    ✓     |
| concatenate            | ✓         |    ✗     |
| convert                | ✓         |    ✓     |
| cosine                 | ✓         |    ✓     |
| divide                 | ✓         |    ✓     |
| dot_general            | ✓         |    ✓     |
| dynamic_slice          | ✓         |    ✗     |
| dynamic_update_slice   | ✓         |    ✗     |
| equal                  | ✓         |    ✓     |
| exp                    | ✓         |    ✓     |
| expm1                  | ✓         |    ✓     |
| fill                   | ✓         |    ✗     |
| floor                  | ✓         |    ✓     |
| greater                | ✓         |    ✓     |
| greater_equal          | ✓         |    ✓     |
| if                     | ✓         |    ✓     |
| iota                   | ✓         |    ✗     |
| is_finite              | ✓         |    ✗     |
| less                   | ✓         |    ✓     |
| less_equal             | ✓         |    ✓     |
| log                    | ✓         |    ✓     |
| log1p                  | ✓         |    ✓     |
| logistic               | ✓         |    ✓     |
| maximum                | ✓         |    ✓     |
| minimum                | ✓         |    ✓     |
| mul                    | ✓         |    ✓     |
| negate                 | ✓         |    ✓     |
| not                    | ✓         |    ✗     |
| not_equal              | ✓         |    ✓     |
| or                     | ✓         |    ✗     |
| pad                    | ✓         |    ✓     |
| popcnt                 | ✓         |    ✗     |
| power                  | ✓         |    ✓     |
| print                  | ✓         |    ✗     |
| reduce_all             | ✓         |    ✗     |
| reduce_any             | ✓         |    ✗     |
| reduce_max             | ✓         |    ✓     |
| reduce_min             | ✓         |    ✓     |
| reduce_prod            | ✓         |    ✗     |
| reduce_sum             | ✓         |    ✓     |
| remainder              | ✓         |    ✗     |
| reshape                | ✓         |    ✓     |
| reverse                | ✓         |    ✓     |
| rng_bit_generator      | ✓         |    ✗     |
| round                  | ✓         |    ✓     |
| rsqrt                  | ✓         |    ✓     |
| select                 | ✓         |    ✓     |
| shift_left             | ✓         |    ✗     |
| shift_right_arithmetic | ✓         |    ✗     |
| shift_right_logical    | ✓         |    ✗     |
| sign                   | ✓         |    ✓     |
| sine                   | ✓         |    ✓     |
| sqrt                   | ✓         |    ✓     |
| static_slice           | ✓         |    ✗     |
| sub                    | ✓         |    ✓     |
| tan                    | ✓         |    ✓     |
| tanh                   | ✓         |    ✓     |
| transpose              | ✓         |    ✓     |
| while                  | ✓         |    ✗     |
| xor                    | ✓         |    ✗     |
