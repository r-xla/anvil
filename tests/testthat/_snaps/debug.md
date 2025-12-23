# printer for debug box

    Code
      DebugBox(AbstractTensor("f32", c(2, 2), TRUE))
    Output
      f32?{2,2}

---

    Code
      DebugBox(AbstractTensor("f32", c(2, 2), FALSE))
    Output
      f32{2,2}

---

    Code
      DebugBox(AbstractTensor("f32", c(), FALSE))
    Output
      f32{}

---

    Code
      DebugBox(ConcreteTensor(nv_tensor(1:4, dtype = "f32", shape = c(2, 2))))
    Output
       1.0000 3.0000
       2.0000 4.0000
      [ CPUf32{2x2} ] 

---

    Code
      DebugBox(LiteralTensor(1, shape = c(2, 3), ambiguous = TRUE))
    Output
      1:f32?{2,3}

---

    Code
      DebugBox(LiteralTensor(1, shape = c(2, 3), ambiguous = FALSE))
    Output
      1:f32{2,3}

---

    Code
      DebugBox(LiteralTensor(1, shape = c(), ambiguous = FALSE))
    Output
      1:f32{}

