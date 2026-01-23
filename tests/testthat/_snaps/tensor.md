# tensor

    Code
      x
    Output
      AnvilTensor
       1
       2
       3
       4
      [ CPUi32{4x1} ] 

# nv_scalar

    Code
      x
    Output
      AnvilTensor
       1
      [ CPUf32{} ] 

# AbstractTensor

    Code
      x
    Output
      AbstractTensor(dtype=f32, shape=2x3) 

# ConcreteTensor

    Code
      x
    Output
      ConcreteTensor
       1 3 5
       2 4 6
      [ CPUf32{2x3} ] 

