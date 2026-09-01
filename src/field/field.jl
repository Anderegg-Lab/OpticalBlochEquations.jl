using LoopVectorization: @turbo
using StructArrays: StructVector

"""
    Field([])
"""
mutable struct Field{T,F1,F2}
    k::SVector{3, T}                        # k-vector
    ϵ::F1                                   # function to update `ϵ` according to the time `t`
    ϵ_val::SVector{3, Complex{T}}           # polarization
    ω::T                                    # frequency
    s_max::T
    s_scalar_func::F2                       # function to calculate saturation parameter
    s::T
    re::T                                   
    im::T
    kr::T                                   # current value of `k ⋅ r`
    E::SVector{3, Complex{T}}               # current value of the field
    ϕ::T                                    # phase of laser
end
function Field(k, ϵ, ω, s_max, s_scalar_func, ϕ)
    _zero = zero(Float64)
    Field{Float64, typeof(ϵ), typeof(s_scalar_func)}(k, ϵ, SVector(_zero,_zero,_zero), ω, s_max, s_scalar_func, 0., _zero, _zero, _zero, SVector(_zero,_zero,_zero), ϕ)
end
function Field(T, k, ϵ, ω, s_max, s_scalar_func, ϕ)
    _zero = zero(T)
    Field{T, typeof(ϵ), typeof(s_scalar_func)}(k, ϵ, SVector(_zero,_zero,_zero), ω, s_max, s_scalar_func, 0., _zero, _zero, _zero, SVector(_zero,_zero,_zero), ϕ)
end
export Field

function update_fields!(fields::StructVector{Field{T,F1,F2}}, r, t) where {T,F1,F2}
    # Fields are represented as ϵ_q * exp(i(kr - ωt)), where ϵ_q is in spherical coordinates
    @inbounds @fastmath for i ∈ eachindex(fields)
        k = fields.k[i]
        fields.kr[i] = k ⋅ r
        fields.ϵ_val[i] = fields.ϵ[i](t)
        fields.s[i] = fields.s_scalar_func[i](r,t) * fields.s_max[i]
    end
    @turbo for i ∈ eachindex(fields)
        fields.im[i], fields.re[i] = sincos(- fields.kr[i] + fields.ω[i] * t + fields.ϕ[i])
    end
    @inbounds @fastmath for i ∈ eachindex(fields)
        val = (fields.re[i] + im * fields.im[i]) .* conj.(fields.ϵ_val[i])
        fields.E[i] = val
    end
    return nothing
end
export update_fields!