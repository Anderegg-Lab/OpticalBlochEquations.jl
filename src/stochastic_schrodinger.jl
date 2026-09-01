using Distributions, StatsBase

"""
    Collapse wavefunction. Based on [ref].
"""
function stochastic_collapse!(integrator)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = integrator.u
    
    p⁺ = 0.0
    p⁰ = 0.0
    p⁻ = 0.0

    for i ∈ 1:n_excited
        c_i = ψ[n_ground + i]
        for j ∈ 1:n_excited
            c_j = ψ[n_ground + j]
            for k ∈ 1:n_ground
                p⁺ += real(conj(c_i) * c_j * conj(d[k,n_ground+i,1]) * d[k,n_ground+j,1])
                p⁰ += real(conj(c_i) * c_j * conj(d[k,n_ground+i,2]) * d[k,n_ground+j,2])
                p⁻ += real(conj(c_i) * c_j * conj(d[k,n_ground+i,3]) * d[k,n_ground+j,3])
            end
            # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
            # whereas the polarization of the emitted photon is m_g - m_e
        end
    end

    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end
    
    for i in 1:n_ground
        for j in (n_ground+1):n_states
            ψ[i] += ψ[j] * d[i,j,pol]
        end
    end
    
    # zero excited state amplitudes
    for i ∈ (n_ground + 1):n_states
        ψ[i] = 0.0
    end
    
    # is this normalization required?
    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    p.n_scatters += 1
    
    # zero excited state populations
    for i ∈ (n_states+1):(n_states+n_excited)
        integrator.u[i] = 0.0
    end

    time_before_decay = integrator.t - p.last_decay_time
    for i ∈ 1:3
        # kick = rand(Normal(0, sqrt( 2p.diffusion_constant[i] * time_before_decay )))
        # integrator.u[p.n_states + p.n_excited + 3 + i] += kick / p.mass
        kick = sqrt( 2p.diffusion_constant[i] * time_before_decay )
        integrator.u[p.n_states + p.n_excited + 3 + i] += rand((-1,1)) * kick / p.mass
    end
    p.last_decay_time = integrator.t

    p.time_to_decay = rand(p.decay_dist)

    return nothing
end
export stochastic_collapse!

"""
    extra_p should contain n_excited

    ψ in the output will be of the following format:
    the first n_states indicies will be the coefficients of the current state;
    the next n_excited indicies is the time-integrated excited state population (reset by callbacks);
    the next 3 indicies are the current position;
    the next 3 indicies are the current velocity;
    the last 3 indicies are the current force.

    Units are:
    time -> 1/Γ
    position -> 1/k
    velocity -> Γ/k
    energies -> 1/Γ
"""
function schrodinger_stochastic(
    particle, states, fields, d, ψ₀, mass, n_excited;
    params=nothing, λ=1.0, Γ=2π, update_H_and_∇H=update_H_and_∇H, diffusion_constant=[0.,0.,0.])

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    fields = StructArray(fields)

    k = 2π / λ
    
    # time unit: 1/Γ
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = ComplexF64

    H = StructArray( zeros(type_complex, n_states, n_states) )
    H₀ = deepcopy(H)
    ∇H = SVector{3, ComplexF64}(0,0,0)

    ∇H_x = deepcopy(H)
    ∇H_y = deepcopy(H)
    ∇H_z = deepcopy(H)

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    Js = Array{Jump}(undef, 0)
    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]
    for s′ ∈ eachindex(states), s ∈ s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            J = Jump(s, s′, q, dme)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    ψ_soa = StructArray(ψ₀)
    dψ_soa = StructArray(ψ₀)
    
    # ψ contains the state vector, accumulated excited state populations, position, velocity, force, state vector for f * ψ
    ψ = zeros(ComplexF64, n_states + n_excited + 3 + 3 + 3 + n_states)
    ψ[1:n_states] .= ψ₀
    ψ[n_states + n_excited + 1: n_states + n_excited + 3] .= r
    ψ[n_states + n_excited + 4: n_states + n_excited + 6] .= v
    dψ = deepcopy(ψ)

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    # E = @MVector Complex{Float64}[0,0,0]
    # E_k = [@MVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    decay_dist = Exponential(1)

    # Compute the non-Hermitian part of the Hamiltonian
    H_nh = StructArray(zeros(ComplexF64, n_states, n_states))
    for i ∈ eachindex(Js)
        J = Js[i]
        for j ∈ i:length(Js)
            J′ = Js[j]
            if (J.s′ == J′.s′) && (J.q == J′.q)
                H_nh[J.s, J′.s] -= (im/2) * conj(J.r) * J′.r
            end
        end
    end

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        H=H, H₀=H₀, ∇H=∇H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, 
        λ=λ, k=k, Γ=Γ,
        E=E, E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        params=params, mass = mass, update_H_and_∇H=update_H_and_∇H, populations = zeros(Float64, n_states),
        n_scatters=0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{ComplexF64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        last_decay_time=0.0,
        ∇H_x=∇H_x,
        ∇H_y=∇H_y,
        ∇H_z=∇H_z,
        ∇H_x_ψ=deepcopy(ψ_soa),
        ∇H_y_ψ=deepcopy(ψ_soa),
        ∇H_z_ψ=deepcopy(ψ_soa),
        H_nh=H_nh,
        diffusion_constant=diffusion_constant
        )

    return p
end
export schrodinger_stochastic

function ψ_stochastic_repump!(dψ, ψ, p, t)
    if ~p.is_dark
        ψ_stochastic_potential!(dψ, ψ, p, t)
    else
        for i ∈ 1:(p.n_states + p.n_excited)
            dψ[i] = 0.0
        end
        for i ∈ (p.n_states + p.n_excited + 4):(p.n_states + p.n_excited + 6)
            dψ[i] = 0.0
        end
    end
    return nothing
end
export ψ_stochastic_repump!

function SE_collapse_repump!(integrator)
    # if the collapse condition has been satisfied and the molecule was in a dark state, move it back to a bright state (all states equally likely)
    if integrator.p.is_dark

        integrator.p.is_dark = false
        n_states = length(integrator.p.states)
        n_excited = integrator.p.n_excited

        # excite molecule to a random excited state
        i = Int(floor(rand()*n_excited)) + 1
        for i ∈ 1:n_states
            integrator.u[i] = 0.0
        end
        integrator.u[n_states - n_excited + i] = 1.0

        # decay back down to the ground state with a single photon recoil
        SE_collapse_pol_always_repump!(integrator, false, 1)
        
    else
        # scatter; add two photon scatters for diffusion from decay AND absorption
        SE_collapse_pol_always_repump!(integrator, true, 2)

        # decay to a dark state
        if rand() > integrator.p.FC_mainline
            integrator.p.is_dark = true
            integrator.p.dark_time = rand(integrator.p.dark_time_dist)
            integrator.p.dark_time_t0 = integrator.t
        end
    end
end
export SE_collapse_repump!

function condition(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = 0.0
    for i ∈ 1:p.n_excited
        p_i = real(u[p.n_states+i])
        integrated_excited_pop += p_i
    end
    _condition = integrated_excited_pop - p.time_to_decay
    return _condition
end
export condition

function condition_repump(u, t, integrator)
    p = integrator.p
    _condition = 0.0
    if integrator.p.is_dark
        _condition += (t - p.dark_time_t0) - p.dark_time
    else
        integrated_excited_pop = 0.0
        @inbounds @fastmath for i ∈ 1:p.n_excited
            integrated_excited_pop += real(u[p.n_states+i])
        end
        _condition += integrated_excited_pop - p.time_to_decay
    end

    # terminate if particle is too far from center
    # r = 0.0
    # for i ∈ 1:3
    #     r += norm(u[p.n_states + p.n_excited + i])^2
    # end
    # r = sqrt(r)
    # if r >= 2e-3 * integrator.p.k
    #    terminate!(integrator)
    # end

    return _condition
end
export condition_repump

function ψ_stochastic_potential!(dψ, ψ, p, t)
    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, mass, k, Γ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    base_to_soa!(ψ, ψ_soa)
    
    update_H!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2, Js) # molecule-light Hamiltonian in schrodinger picutre
    
    update_eiωt!(eiωt, ω, t)
    Heisenberg!(H, eiωt)  # molecule-light Hamiltonian in interation picture
    
    ∇H = p.update_H_and_∇H(H₀, p, r, t) # Zeeman and ODT hamiltonian in schrodinger picutre
    Heisenberg!(H₀, eiωt) # Zeeman and ODT Hamiltonian in interaction picture
    
    @turbo for i ∈ eachindex(H)
        H.re[i] += H₀.re[i]
        H.im[i] += H₀.im[i]
    end
    
    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    soa_to_base!(dψ, dψ_soa)
    
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt) # force due to lasers

    H₀_expectation = operator_matrix_expectation(H₀, ψ_soa)
    f += ∇H .* (-H₀_expectation) # force due to conservative potential

    # add gravity to the force
    g = -9.81 / (Γ^2/k)
    f += SVector{3,Float64}(0,mass*g,0)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
    end

    # for i ∈ 1:3
    #     kick = sqrt( 2p.diffusion_constant[i] * p.dt )
    #     ψ[n_states + n_excited + 3 + i] += rand(Normal(0,kick)) / p.mass
    # end

    return nothing
end
export ψ_stochastic_potential!

""" 
    Compute the expectation value ⟨ψ|O|ψ⟩ for an operator O and a state vector |ψ⟩.
"""
function operator_matrix_expectation(O, state)
    O_re = zero(Float64)
    @turbo for i ∈ eachindex(state)
        re_i = state.re[i]
        im_i = state.im[i]
        for j ∈ eachindex(state)
            re_j = state.re[j]
            im_j = state.im[j]
            cicj_re = re_i * re_j + im_i * im_j # real part of ci * cj
            cicj_im = re_i * im_j - im_i * re_j
            O_re += O.re[i,j] * cicj_re - O.im[i,j] * cicj_im
        end
    end
    return O_re
end
export operator_matrix_expectation

""" 
    Compute the expectation value ⟨ψ|O²|ψ⟩ for an operator O and a state vector |ψ⟩.
"""
function operator_sq_matrix_expectation(O, state)
    O_re = zero(Float64)
    @turbo for i ∈ eachindex(state)
        re_i = state.re[i]
        im_i = state.im[i]
        for j ∈ eachindex(state)
            re_j = state.re[j]
            im_j = state.im[j]
            cicj_re = re_i * re_j + im_i * im_j # real part of ci * cj
            cicj_im = re_i * im_j - im_i * re_j
            O_re += O.re[i,j] * cicj_re - O.im[i,j] * cicj_im
        end
    end
    return O_re
end
export operator_matrix_expectation

function operator_matrix_expectation_column(O, state)
    O_re = zero(Float64)
    @turbo for j ∈ eachindex(state)
        re_j = state.re[j]
        im_j = state.im[j]
        for i ∈ eachindex(state)
            re_i = state.re[i]
            im_i = state.im[i]
            cicj_re = re_i * re_j + im_i * im_j # real part of ci * cj
            cicj_im = re_i * im_j - im_i * re_j
            O_re += O.re[i,j] * cicj_re - O.im[i,j] * cicj_im
        end
    end
    return O_re
end
export operator_matrix_expectation_column

function operator_to_matrix_zero_padding2(OA, A_states, B_states)
    """
    OA is an operator on Hilbert space A (basis = A_states).
    We would like to extend A to the direct-sum space A ⨁ B by padding with zeros, i.e.
    <i|OAB|j> = 0 if i∉A or j∉A, <i|OAB|j> = <i|OA|j> if i∈A and j∈A.
    """
    n_A = length(A_states)
    n_B = length(B_states)
    OAB_mat = zeros(ComplexF64, n_A+n_B, n_A+n_B)
    OAB_mat[1:n_A, 1:n_A] .= operator_to_matrix(OA, A_states)
    return OAB_mat
end
export operator_to_matrix_zero_padding2