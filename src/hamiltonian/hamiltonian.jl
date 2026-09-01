function update_H!(H, p, r, τ)
    @turbo for i in eachindex(H)
        H.re[i] = 0.0 #p.H₀.re[i]
        H.im[i] = 0.0 #p.H₀.im[i]
    end
    return nothing
end

function set_H_zero!(H)
    @turbo for i ∈ eachindex(H)
        H.re[i] = zero(eltype(H))
        H.im[i] = zero(eltype(H))
    end
    return nothing
end
export set_H_zero!

function set_H_to_H′!(H, H′)
    @turbo for i ∈ eachindex(H)
        H.re[i] = H′.re[i]
        H.im[i] = H′.im[i]
    end
    return nothing
end
export set_H_to_H′!

function update_H!(p, τ, r, fields, H, E_k, ds, ds_state1, ds_state2, Js)
 
    set_H_zero!(H)

    update_fields!(fields, r, τ)

    # Set summed fields to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # Sum updated fields
    @inbounds for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    @inbounds @fastmath for q ∈ 1:3
        E_q = p.E[q]
        E_q_re = real(E_q)
        E_q_im = imag(E_q)
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            val_re = E_q_re * d_re - E_q_im * d_im
            val_im = E_q_re * d_im + E_q_im * d_re
            H.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d⋅E
            H.im[n,m] += -val_im
            H.re[m,n] += -val_re
            H.im[m,n] -= -val_im
        end
    end

    # diagonal terms like |e1><e1| for the non-hermitian part of H
    @inbounds for J ∈ Js
        a = real(J.r)
        b = imag(J.r)
        r_squared = a^2 + b^2
        H.im[J.s, J.s] -= r_squared / 2
    end

    # # off-diagonal terms like |e1><e2| for the non-hermitian part of H
    # # they only exist when the ground state of the jump operator is the same (and have the same polarization)
    # @inbounds for i ∈ eachindex(Js)
    #     @inbounds for j ∈ (i+1):length(Js)
    #         J′ = Js[j]

    #         if (J.s′ == J′.s′) && (J.q == J′.q)
    #             H.im[J.s, J′.s] -= conj(J.r) * sqrt(J′.r)
    #         end
    # end

    # @inbounds for i ∈ eachindex(Js)
    #     J = Js[i]
    #     dρ_soa[J.s′, J.s′] += norm(J.r)^2 * ρ_soa[J.s, J.s]
    #     @inbounds for j ∈ (i+1):length(Js)
    #         J′ = Js[j]
    #         if J.q == J′.q
    #             val = conj(J.r) * J′.r * ρ_soa[J.s, J′.s]
    #             dρ_soa[J.s′, J′.s′] += val
    #             dρ_soa[J′.s′, J.s′] += conj(val)
    #         end
    #     end
    # end

    return nothing
end
export update_H!

function update_H_schrodinger!(p, τ, r, fields, H, E_k, ds, ds_state1, ds_state2)
 
    set_H_zero!(H)

    update_fields!(fields, r, τ)

    # Set summed fields to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # Sum updated fields
    @inbounds for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    @inbounds @fastmath for q ∈ 1:3
        E_q = p.E[q]
        E_q_re = real(E_q)
        E_q_im = imag(E_q)
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            val_re = E_q_re * d_re - E_q_im * d_im
            val_im = E_q_re * d_im + E_q_im * d_re
            H.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d⋅E
            H.im[n,m] += -val_im
            H.re[m,n] += -val_re
            H.im[m,n] -= -val_im
        end
    end

    return nothing
end
export update_H_schrodinger!

function update_H_obes!(p, τ, r, H₀, fields, H, E_k, ds, ds_state1, ds_state2, Js)

    set_H_to_H′!(H, H₀)

    update_fields!(fields, r, τ)

    # Set summed fields to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # Sum updated fields
    @inbounds @fastmath for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    @inbounds @fastmath for q ∈ 1:3
        E_q = p.E[q]
        E_q_re = real(E_q)
        E_q_im = imag(E_q)
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            d_im = -d_im # added 8/23/24, making sure that Hamiltonian is -d*⋅E for cases where d is complex
            val_re = E_q_re * d_re - E_q_im * d_im
            val_im = E_q_re * d_im + E_q_im * d_re
            H.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d*⋅E
            H.im[n,m] += -val_im
            H.re[m,n] += -val_re
            H.im[m,n] -= -val_im
        end
    end

    @inbounds for J ∈ Js
        a = real(J.r)
        b = imag(J.r)
        r_squared = a^2 + b^2
        H.im[J.s, J.s] -= r_squared / 2
    end

    return nothing
end
export update_H_obes!