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
       1.0000
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
      ConcreteTensor(dtype=f32, shape=2x3) 
       1.0000 3.0000 5.0000
       2.0000 4.0000 6.0000
      [ CPUf32{2x3} ] 

