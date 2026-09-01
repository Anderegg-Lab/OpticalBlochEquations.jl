"""
    Notes: 
    -   H is in the interaction picture (using function `Heisenberg!`)
    -   ψ is in the interaction picture, because H is in the interaction picture

    Note that the wavefunction is always in the interaction picture.
    (Except for ψ_soa)
    
    All operators are in the Schrodinger picture
"""
function ψ_stochastic!(dψ, ψ, p, t)

    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, mass, k, Γ = p

    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(
        real(ψ[n_states + n_excited + 1]),
        real(ψ[n_states + n_excited + 2]), 
        real(ψ[n_states + n_excited + 3])
        )

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    update_eiωt!(eiωt, ω, t)

    base_to_soa!(ψ, ψ_soa)
    Heisenberg_turbo_state!(ψ_soa, eiωt, -1)

    update_H_dipole!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2, Js) # molecule-light Hamiltonian in Schrodinger picutre
    
    ∇H = p.update_H_and_∇H(H₀, p, r, t) # Zeeman and ODT Hamiltonian in Schrodinger picutre

    # add the Zeeman and ODT Hamiltonian to dipole Hamiltonian
    @turbo for i ∈ eachindex(H)
        H.re[i] += H₀.re[i]
        H.im[i] += H₀.im[i]
    end 

    mul_turbo!(p.∇H_x_ψ, p.∇H_x, ψ_soa)
    mul_turbo!(p.∇H_y_ψ, p.∇H_y, ψ_soa)
    mul_turbo!(p.∇H_z_ψ, p.∇H_z, ψ_soa)

    fx_exp = state_overlap_real(ψ_soa, p.∇H_x_ψ)
    fy_exp = state_overlap_real(ψ_soa, p.∇H_y_ψ)
    fz_exp = state_overlap_real(ψ_soa, p.∇H_z_ψ)

    f = SVector(fx_exp, fy_exp, fz_exp)

    # # add force due to conservative potential
    # H₀_expectation = operator_matrix_expectation(H₀, ψ_soa)
    # f += SVector(-∇H[1] * H₀_expectation, -∇H[2] * H₀_expectation, -∇H[3] * H₀_expectation)

    # add gravity to the force
    g = -9.81 / (Γ^2/k)
    f += SVector{3,Float64}(0,mass*g,0)

    # calculate change in the state vector
    mul_turbo_im_minus!(dψ_soa, H, ψ_soa)
    Heisenberg_turbo_state!(dψ_soa, eiωt, +1)
    soa_to_base!(dψ, dψ_soa)
    
    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
    end

    # update force
    for i ∈ 1:3
        ψ[n_states + n_excited + 6 + i] = f[i]
    end

    return nothing
end
export ψ_stochastic!


function schrodinger_stochastic_constant_diffusion(
    particle, states, fields, d, ψ₀, mass, n_excited, diffusion_constant;
    params=nothing, λ=1.0, Γ=2π, update_H_and_∇H=update_H_and_∇H, dt=0.0)
    """
    extra_p should contain n_excited
    
    ψ in the output will be of the following format:
    the first n_states indicies will be the coefficients of the current state;
    the next n_excited indicies is the time-integrated excited state population (reset by callbacks);
    the next 3 indicies are the current position;
    the next 3 indicies are the current velocity;
    the last 3 indicies are the current force.
    """

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

    H = StructArray( zeros(type_complex, n_states, n_states) ) # light-molecule (dipole) Hamiltonian
    H₀ = deepcopy(H) # Zeeman and ODT Hamiltonian

    
    ∇H = SVector{3, ComplexF64}(0,0,0) # gradient of the ODT Hamiltonian = ∇H * H_ODT. ∇H is just a 3-vector

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
    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            rate = norm(dme)^2 / 2
            J = Jump(s, s′, q, rate)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    ψ_soa = StructArray(ψ₀)
    dψ_soa = StructArray(ψ₀)
    
    # ψ contains the state vector, accumulated excited state populations, position, velocity, force
    ψ = zeros(ComplexF64, n_states + n_excited + 3 + 3 + 3)
    ψ[1:n_states] .= ψ₀
    ψ[n_states + n_excited + 1: n_states + n_excited + 3] .= r
    ψ[n_states + n_excited + 4: n_states + n_excited + 6] .= v
    dψ = deepcopy(ψ)

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    decay_dist = Exponential(1)

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        H=H, H₀=H₀, ∇H=∇H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, 
        λ=λ, k=k, Γ=Γ,
        E=E, E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        params=params, mass = mass, update_H_and_∇H=update_H_and_∇H, populations = zeros(Float64, n_states),
        n_scatters = 0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{ComplexF64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        last_decay_time=0.0,
        diffusion_constant=diffusion_constant,
        dt=dt
        )

    return p
end
export schrodinger_stochastic_constant_diffusion