# OnionStyle

Syntactic sugar for the Onion neural network library.

## Exports

- **`⁻¹`, `ᵀ`, `ᴴ`** — Postfix operators for inverse, transpose, and hermitian. Used as `(A)ᵀ`.

- **`→`** — Type conversion operator. `x → T` converts `x` to type `T`. Supports partial application: `→ Float32` returns a converter.

- **`Optional{T}`** — Alias for `Union{T, Nothing}`.

- **`@staticmap`** — Compile-time mapping function with `===` semantics. Supports block, comma-separated, `{}` grouping, and `_` fallback:

  ```julia
  f = @staticmap begin
      {a, b} => 1
      c => 2
      _ => 0
  end
  ```
