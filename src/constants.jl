const cart2sph = @SMatrix [
    +1/√2 +im/√2 0;
    0 0 1
    -1/√2 +im/√2 0;
]
export cart2sph

const sph2cart = inv(cart2sph)
export sph2cart

const x̂ = SVector{3, Float64}(1,0,0)
const ŷ = SVector{3, Float64}(0,1,0)
const ẑ = SVector{3, Float64}(0,0,1)
export x̂, ŷ, ẑ
const ê = [x̂, ŷ, ẑ]
export ê

const ϵ₊ = SVector{3, ComplexF64}(-1/√2, -im/√2, 0)
const ϵ₋ = SVector{3, ComplexF64}(+1/√2, -im/√2, 0)
const ϵ₀ = SVector{3, ComplexF64}(0.0, 0.0, 1.0)
const ϵ_cart = [ϵ₋, ϵ₀, ϵ₊]
export ϵ₊, ϵ₋, ϵ₀
export ϵ_cart

const σ⁻ = SVector{3, ComplexF64}(1.0, 0.0, 0.0)
const σ⁺ = SVector{3, ComplexF64}(0.0, 0.0, 1.0)
const σ⁰ = SVector{3, ComplexF64}(0.0, 1.0, 0.0)
export σ⁻, σ⁺, σ⁰

const qs = @SVector [-1, 0, 1]
export qs