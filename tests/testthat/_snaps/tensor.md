# tensor

    Code
      x
    Output
      AnvilTensor
       1
       2
       3
       4
      [ CPUi32{4,1} ] 

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
      [ CPUf32{2,3} ] 

# stablehlo dtype is printed

    Code
      nv_tensor(TRUE)
    Output
      AnvilTensor
       1
      [ CPUi1{1} ] 

