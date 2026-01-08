# inline_scalarish_constants: replaces all references to converted constants

    Code
      graph
    Output
      <Graph>
        Inputs:
          %x1: f32[]
        Constants:
          %c1: f32[]
        Body:
          %1: f32[] = add(%x1, %c1)
          %2: f32[] = mul(%1, %c1)
        Outputs:
          %2: f32[] 

