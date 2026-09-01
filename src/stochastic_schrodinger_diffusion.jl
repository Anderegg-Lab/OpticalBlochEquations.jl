function update_H_dipole!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2, Js)
    
    # reset matrices
    set_H_to_H′!(H, p.H_nh)

    set_H_zero!(p.∇H_x)
    set_H_zero!(p.∇H_y)
    set_H_zero!(p.∇H_z)

    # Reset total E field and E dot k to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # update each laser at the current time and position
    update_fields!(fields, r, t)
    
    # Calculate total E field and total E dot k
    @inbounds for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    # calculate dipole Hamiltonian matrix elements
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
    
    # calculate matrix elements of the gradient of the dipole Hamiltonian
    calculate_grad_H_all!(p, E_k, ds, ds_state1, ds_state2)

    return nothing
end
export update_H_dipole!

function calculate_grad_H_all!(p, E_k, ds, ds_state1, ds_state2)

    ∇H = (p.∇H_x, p.∇H_y, p.∇H_z)

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]

        for k ∈ 1:3
            E_kq = E_k[k][q]
            E_kq_re = -imag(E_kq) # note: multiply by -i
            E_kq_im = +real(E_kq) # note: multiply by -i

            ∇H_k = ∇H[k]

            for i ∈ eachindex(ds_q)
                m = ds_state1_q[i] # excited state
                n = ds_state2_q[i] # ground state
                d_re = ds_q_re[i]
                d_im = ds_q_im[i]

                val_x_re = E_kq_re * d_re - E_kq_im * d_im
                val_x_im = E_kq_re * d_im + E_kq_im * d_re

                ∇H_k.re[n,m] += -val_x_re # minus sign to make sure Hamiltonian is -d⋅E
                ∇H_k.im[n,m] += -val_x_im
                ∇H_k.re[m,n] += -val_x_re
                ∇H_k.im[m,n] -= -val_x_im

            end
        end
    end

    return nothing
end
export calculate_grad_H_all!

function calculate_grad_H!(∇H_k, k, p, E_k, ds, ds_state1, ds_state2)
    E_along_k = E_k[k]
    @inbounds @fastmath for q ∈ 1:3
        E_kq = E_along_k[q]
        E_kq_re = +imag(E_kq) # note: multiply by -i
        E_kq_im = -real(E_kq) # note: multiply by -i
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
            val_re = E_kq_re * d_re - E_kq_im * d_im
            val_im = E_kq_re * d_im + E_kq_im * d_re
            ∇H_k.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d⋅E
            ∇H_k.im[n,m] += -val_im
            ∇H_k.re[m,n] += -val_re
            ∇H_k.im[m,n] -= -val_im
        end
    end
    return nothing
end
export calculate_grad_H!