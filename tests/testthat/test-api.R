describe("nv_concatenate", {
  it("auto-promotes to common", {
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(c(1, 2)), nv_tensor(3:4))),
      nv_tensor(c(1, 2, 3, 4))
    )
  })
  it("can concatenate literals", {
    expect_equal(
      jit_eval(nv_concatenate(1L, 2L)),
      nv_tensor(1:2)
    )
    expect_equal(
      jit_eval(nv_concatenate(1L, 2L, dimension = 1L)),
      nv_tensor(1:2)
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1:2), 3L)),
      nv_tensor(1:3)
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1L), 2L)),
      nv_tensor(1:2)
    )
  })
  it("fails when dimension is out of bounds", {
    expect_error(
      jit_eval(nv_concatenate(nv_tensor(1:2, shape = c(2, 1)), nv_tensor(3:4, shape = c(2, 1)), dimension = 3L))
    )
  })
  it("can concatenate 2d tensors", {
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1:2, shape = c(2, 1)), nv_tensor(3:4, shape = c(2, 1)), dimension = 1L)),
      nv_tensor(1:4, shape = c(4, 1))
    )
    expect_equal(
      jit_eval(nv_concatenate(nv_tensor(1:2, shape = c(2, 1)), nv_tensor(3:4, shape = c(2, 1)), dimension = 2L)),
      nv_tensor(1:4, shape = c(2, 2), dtype = "i32")
    )
  })
  it("fails with incompatible shapes", {
    expect_error(
      jit_eval(nv_concatenate(nv_tensor(1, shape = c(1, 1, 1)), nv_tensor(2, shape = c(1, 1)), dimension = 1L))
    )
  })
})
