# Convert an `S7` class to a mutable `S7` object

Convert an `S7` class to a mutable `S7` object

## Usage

``` r
mut(x)
```

## Arguments

- x:

  `S7_class` an `S7_class` class constructor - the function you would
  otherwise call to create a new object of that class.

## Examples

``` r
library(S7)

class_ex <- new_class(
  "class_example",
  properties = list(
    # validators
    value = new_property(
      class = class_integer,
      validator = function(value) {
        if (value == 0L) "value cannot be exactly 0"
      }
    ),
    # read-only properties
    is_gt_0 = new_property(
      class = class_logical,
      getter = function(self) {
        self@value > 0
      }
    )
  )
)

# make a mutable version of our class
ex <- mut(class_ex)(value = 3L)
ex@value
#> [1] 3

# we can make a copy and update our value property
ex_ref <- ex
ex_ref@value <- 30L

# all values reference the same data, our original is updated
ex@value
#> [1] 30
```
