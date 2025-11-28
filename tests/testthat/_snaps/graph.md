# Graph: printing

    Code
      graph
    Output
      <Graph>
        Inputs:
          %x1: f32[]
          %x2: f32[]
        Body:
          %1: f32[] = add(%x1, %x2)
        Outputs:
          %1: f32[] 

---

    Code
      graph1
    Output
      <Graph>
        Inputs:
          %x1: f32[2, 5]
        Constants:
          %c1: f32[]
        Body:
          %1: f32[] = sum [dims = c(1, 2), drop = TRUE] (%x1)
          %2: f32[] = divide(%1, %c1)
        Outputs:
          %2: f32[] 

