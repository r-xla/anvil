# printer for debug box

    Code
      DebugBox(AbstractArray("f32", c(2, 2), TRUE))
    Output
      f32?{2,2}

---

    Code
      DebugBox(AbstractArray("f32", c(2, 2), FALSE))
    Output
      f32{2,2}

---

    Code
      DebugBox(AbstractArray("f32", c(), FALSE))
    Output
      f32{}

---

    Code
      DebugBox(LiteralArray(1, shape = c(2, 3), ambiguous = TRUE))
    Output
      1:f32?{2,3}

---

    Code
      DebugBox(LiteralArray(1, shape = c(2, 3), ambiguous = FALSE))
    Output
      1:f32{2,3}

---

    Code
      DebugBox(LiteralArray(1, shape = c(), ambiguous = FALSE))
    Output
      1:f32{}

---

    Code
      DebugBox(ConcreteArray(nv_array(1:4, dtype = "f32", shape = c(2, 2))))
    Output
      DebugBox(ConcreteArray)
       1 3
       2 4
      [ CPUf32{2,2} ] 

