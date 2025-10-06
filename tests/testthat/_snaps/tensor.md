# tensor

    Code
      x
    Output
      AnvilTensor{4x1xi32} 
       1
       2
       3
       4

# nv_scalar

    Code
      x
    Output
      AnvilTensor{f32} 
       1.0000

# ShapedTensor

    Code
      x
    Output
      ShapedTensor(dtype=f32, shape=2x3) 

# ConcreteTensor

    Code
      x
    Output
      ConcreteTensor(dtype=f32, shape=2x3) 
       1.0000 3.0000 5.0000
       2.0000 4.0000 6.0000

