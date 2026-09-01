"""
    Initialize a problem for a stochastic Schrödinger simulation.

    ω0s:            energies of states (angular units)
    ωs:             laser frequency components (angular units)
    sats:           saturation parameters for each frequency component
    pols:           polarizations of laser frequency components
    ψ0:             initial value for ψ
    d:              transition dipole array
    m:              mass of particle
    Γ:              linewidth of transition
    k:              k-value of transition
    params:         custom parameters to be passed to the solver
    add_terms_dψ:   function to add custom terms to dψ, with signature `add_terms_dψ(dψ, ψ, p, r, t)``
"""
function initialize_prob(
        sim_type,
        ω0s,
        ωs,
        sats,
        pols,
        beam_radius,
        d,
        m,
        Γ,
        k,
        sim_params,
        update_params,
        add_terms_dψ;
        freq_groups=nothing,
        iris_factor=Inf
    )

    # get some initial constants
    n_states = length(ω0s)
    n_g = find_n_g(d)
    n_excited = n_states - n_g
    n_freqs = length(ωs)

    # set the integer type for the simulation
    intT = get_int_type(sim_type)

    denom = sim_type((beam_radius*k)^2/2)

    eiω0ts = zeros(Complex{sim_type},n_states)

    k_dirs = 6

    as = zeros(Complex{sim_type},k_dirs,n_freqs)
    ϕs = zeros(sim_type,3,3+n_freqs)
    rs = zeros(sim_type,2,3)
    kEs = zeros(Complex{sim_type}, k_dirs, 3)

    idxs = intT.(reshape(collect(1:(3n_freqs)),3,n_freqs))

    # define polarization array
    ϵs = zeros(Complex{sim_type},k_dirs,n_freqs,3)
    for i ∈ eachindex(pols)
        pol = pols[i]
        ϵs[1,i,:] .= rotate_pol(pol, x̂)
        ϵs[2,i,:] .= rotate_pol(pol, ŷ)
        ϵs[3,i,:] .= rotate_pol(flip(pol), ẑ)
        ϵs[4,i,:] .= rotate_pol(pol, -x̂)
        ϵs[5,i,:] .= rotate_pol(pol, -ŷ)
        ϵs[6,i,:] .= rotate_pol(flip(pol), -ẑ)
    end

    # arrays related to energies
    as = StructArray(MMatrix{k_dirs,n_freqs}(as))
    ωs = MVector{size(ωs)...}(ωs)
    ϕs = MMatrix{size(ϕs)...}(ϕs)
    rs = MMatrix{size(rs)...}(rs)
    kEs = StructArray(MMatrix{size(kEs)...}(kEs))
    ϵs = StructArray(MArray{Tuple{k_dirs,n_freqs,3}}(ϵs))
    idxs = MMatrix{size(idxs)...}(idxs)

    # arrays related to state energies
    ω0s = MVector{size(ω0s)...}(ω0s)
    eiω0ts = StructArray(MVector{size(eiω0ts)...}(eiω0ts))

    ψ = zeros(Complex{sim_type}, n_states)
    ψ = MVector{size(ψ)...}(ψ)
    ψ = StructArray(ψ)

    dψ = deepcopy(ψ)

    ψ_q = deepcopy(ψ)
    ψ_q = MArray{Tuple{size(ψ)...,3}}(zeros(Complex{sim_type},size(ψ)...,3))
    ψ_q = StructArray(ψ_q)

    E_total = zeros(Complex{sim_type},3)
    E_total = MVector{size(E_total)...}(E_total)
    E_total = StructArray(E_total)

    # note that we take the negative to ensure that the Hamiltonian is -d⋅E
    d_ge = sim_type.(real.(-d[1:n_g,(n_g+1):n_states,:]))
    d_ge = MArray{Tuple{size(d_ge)...}}(d_ge)
    d_eg = permutedims(d_ge,(2,1,3))

    F = MVector{3,sim_type}(zeros(3))

    d = MArray{Tuple{n_states,n_states,3}}(sim_type.(real.(d)))

    d_exp = zeros(Complex{sim_type},3)
    d_exp = MVector{size(d_exp)...}(d_exp)
    d_exp = StructArray(d_exp)

    d_exp_split = zeros(Complex{sim_type},3,2)
    d_exp_split = MMatrix{size(d_exp_split)...}(d_exp_split)
    d_exp_split = StructArray(d_exp_split)

    r = MVector{3}(zeros(sim_type,3))
    r_idx = 2n_states + n_excited
    v_idx = r_idx + 3
    F_idx = v_idx + 3

    decay_dist = Exponential(one(sim_type))
    last_decay_time = zero(sim_type)

    diffusion_constant = MVector{3,sim_type}(zeros(3))
    add_spontaneous_decay_kick = false

    n_scatters = zero(sim_type)

    # Per-group arrays for RWA frequency grouping
    if freq_groups !== nothing
        _use_fg = true
        _g1_f_start = intT(first(freq_groups[1][1]))
        _g1_f_end   = intT(last(freq_groups[1][1]))
        _g1_g_start = intT(first(freq_groups[1][2]))
        _g1_g_end   = intT(last(freq_groups[1][2]))
        _g2_f_start = intT(first(freq_groups[2][1]))
        _g2_f_end   = intT(last(freq_groups[2][1]))
        _g2_g_start = intT(first(freq_groups[2][2]))
        _g2_g_end   = intT(last(freq_groups[2][2]))
    else
        _use_fg = false
        _g1_f_start = _g1_f_end = _g1_g_start = _g1_g_end = intT(0)
        _g2_f_start = _g2_f_end = _g2_g_start = _g2_g_end = intT(0)
    end
    _E_g1 = StructArray(MVector{3}(zeros(Complex{sim_type}, 3)))
    _E_g2 = StructArray(MVector{3}(zeros(Complex{sim_type}, 3)))
    _kEs_g1 = StructArray(MMatrix{6,3}(zeros(Complex{sim_type}, 6, 3)))
    _kEs_g2 = StructArray(MMatrix{6,3}(zeros(Complex{sim_type}, 6, 3)))
    _d_exp_g1 = StructArray(MVector{3}(zeros(Complex{sim_type}, 3)))
    _d_exp_g2 = StructArray(MVector{3}(zeros(Complex{sim_type}, 3)))

    u0 = sim_type.([zeros(n_states)..., zeros(n_states)..., zeros(n_excited)..., zeros(3)..., zeros(3)..., zeros(3)..., zeros(3)...])
    u0[1] = 1.0

    p = MutableNamedTuple(
        u0=u0,
        Γ=Γ,
        ωs=ωs,
        ω0s=ω0s,
        eiω0ts=eiω0ts,
        ϕs=ϕs,
        as=as,
        rs=rs,
        kEs=kEs,
        E_total=E_total,
        ϵs=ϵs,
        idxs=idxs,
        denom=denom,
        iris_factor=sim_type(iris_factor),
        ψ=ψ,
        dψ=dψ,
        ψ_q=ψ_q,
        sim_params=sim_params,
        d_ge=d_ge,
        d_eg=d_eg,
        F=F,
        d=d,
        d_exp=d_exp,
        d_exp_split=d_exp_split,
        r=r,
        r_idx=r_idx,
        v_idx=v_idx,
        F_idx=F_idx,
        n_g=n_g,
        n_excited=n_excited,
        n_states=n_states,
        m=m,
        add_terms_dψ=add_terms_dψ,
        update_params=update_params,
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        last_decay_time=last_decay_time,
        n_scatters=n_scatters,
        diffusion_constant=diffusion_constant,
        add_spontaneous_decay_kick=add_spontaneous_decay_kick,
        sats=sats,
        k=k,
        use_freq_groups=_use_fg,
        g1_f_start=_g1_f_start, g1_f_end=_g1_f_end,
        g1_g_start=_g1_g_start, g1_g_end=_g1_g_end,
        g2_f_start=_g2_f_start, g2_f_end=_g2_f_end,
        g2_g_start=_g2_g_start, g2_g_end=_g2_g_end,
        E_g1=_E_g1, E_g2=_E_g2,
        kEs_g1=_kEs_g1, kEs_g2=_kEs_g2,
        d_exp_g1=_d_exp_g1, d_exp_g2=_d_exp_g2
    )

    return p
end
export initialize_prob

function make_couplings(H)
    ds_state1 = Int64[]
    ds_state2 = Int64[]
    ds = Float64[]
    for i ∈ axes(H, 1)
        for j ∈ i:size(H, 2)
            if norm(H[i,j]) > 1e-10
                push!(ds_state1, j)
                push!(ds_state2, i)
                push!(ds, H[i,j])
            end
        end
    end
    return (ds_state1, ds_state2, ds)
end


@inline function H_expectation(ψ, H, idxs, scalar)
    acc = 0.0f0
    @inbounds for (i, j) in idxs
        Hij = H[i,j]
        if i == j
            acc += real(conj(ψ[i]) * Hij * ψ[i])  # Diagonal term
        else
            acc += 2f0 * real(conj(ψ[i]) * Hij * ψ[j])  # Use symmetry
        end
    end
    return scalar*acc
end
export operator_expectation_state


@inline function H_func(p)
    # slow
    # update_ODT_center_circle!(p.sim_params, t)
    r = p.r
    ODT_x = p.sim_params.ODT_position[1] * p.k
    ODT_z = p.sim_params.ODT_position[2] * p.k
    ODT_y = 0e-3 * p.k
    ODT_size = p.sim_params.ODT_size .* p.k
    gaussian_trap_scalar = exp(-2(r[1]-ODT_x)^2/ODT_size[1]^2) * exp(-2(r[2]-ODT_y)^2/ODT_size[2]^2) * exp(-2(r[3]-ODT_z)^2/ODT_size[3]^2)
    ∇H = (-4(r[1]-ODT_x) / ODT_size[1]^2, -4(r[2]-ODT_y) / ODT_size[2]^2, -4(r[3]-ODT_z) / ODT_size[3]^2)
    return gaussian_trap_scalar, ∇H
end
# export H_func

@inline function H_func_static(p)
    r = p.r
    ODT_pos = p.sim_params.ODT_position
    ODT_size = p.sim_params.ODT_size
    k = p.k

    # Precompute scaled positions
    r1 = r[1]
    r2 = r[2]
    r3 = r[3]

    # Scale ODT sizes once
    sx = ODT_size[1] * k
    sy = ODT_size[2] * k
    sz = ODT_size[3] * k

    # Precompute squared sizes
    sx2 = sx^2
    sy2 = sy^2
    sz2 = sz^2

    # Gaussian scalar
    gaussian_trap_scalar = exp(-2f0 * r1^2 / sx2) *
                           exp(-2f0 * r2^2 / sy2) *
                           exp(-2f0 * r3^2 / sz2)

    # Gradient
    ∇H = SVector{3,Float32}(
        -4f0 * r1 / sx2,
        -4f0 * r2 / sy2,
        -4f0 * r3 / sz2
    )

    return gaussian_trap_scalar, ∇H
end

@inline function w(z, zr, w0)
    return w0*sqrt(1+(z/zr)^2)
end

@inline function H_func_blue_red(p)
    r = p.r
    ODT_size = p.sim_params.ODT_size
    k = p.k

    x = r[1]
    z = r[2]
    y = r[3]

    w0 = ODT_size[1] * k
    zR = ODT_size[2] * k
    # sz = ODT_size[3] * k

    w02 = w0^2
    wz = w(z, zR, w0)
    wz2 = wz^2
    ρ²=x^2+y^2
    # sz2 = sz^2
    gaussian_trap_scalar = 0.0
    ∇H = SVector{3,Float32}(0.0,0.0,0.0)
    if p.sim_params.blue
        gaussian_trap_scalar = (w02/wz2^2) * 2ρ²*exp(-2ρ²/wz2)+exp(-2(z-zR)^2/w02)+exp(-2(z+zR)^2/w02)

        #Gradient

        ∇H = SVector{3,Float32}(
            4x*(w02/wz2^2)*(1-2ρ²/wz2)*exp(-2*ρ²/wz2),
            8*w02^2*z*ρ²*(ρ²-wz2)/(zR^2*wz2^4)*exp(-2ρ²/wz2)-4(z-zR)/w02*exp(-2(z-zR)^2/w02)-4(z+zR)/w02*exp(-2(z+zR)^2/w02),
            4y*(w02/wz2^2)*(1-2ρ²/wz2)*exp(-2*ρ²/wz2),
        )
    else
        gaussian_trap_scalar = (w02/wz2)*exp(-2ρ²/wz2)

        # Gradient
        ∇H = SVector{3,Float32}(
            -4x*gaussian_trap_scalar/wz2,
            2w02*z*gaussian_trap_scalar*(2ρ²/wz2-1)/(wz2*zR^2),
            -4y*gaussian_trap_scalar/wz2
        )
    end


    return gaussian_trap_scalar, ∇H
end

"""
    Time step update function for stochastic Schrödinger simulation.
"""
function ψ_fast!(du, u, p, t)
    
    normalize_u!(u, p.n_states)

    update_r!(u, p.r, p.r_idx)

    p.update_params(p, p.r, t)

    update_ψ!(p.ψ, u, p.n_states)

    update_fields_fast!(p, p.r, t)
    
    update_eiωt_new!(p.eiω0ts, p.ω0s, t)

    Heisenberg_turbo_state!(p.ψ, p.eiω0ts, -1)

    update_ψq!(p.ψ_q, p.d_ge, p.d_eg, p.ψ, p.n_g)

    update_d_exp!(p.d_exp, p.ψ, p.ψ_q, p.n_g)

    gaussian_trap_scalar = update_force!(p, p.F, p.d_exp, p.kEs)
     
    update_dψ!(p.dψ, p.ψ_q, p.E_total, p.n_states, p.n_g)

    p.add_terms_dψ(p.dψ, p.ψ, p, p.r, t, gaussian_trap_scalar) # custom terms to add to dψ

    Heisenberg_turbo_state!(p.dψ, p.eiω0ts, +1)

    update_du!(du, u, p.dψ, p.ψ, p.n_states, p.n_g, p.r_idx, p.F, p.v_idx, p.F_idx, p.m)

    return nothing
end
export ψ_fast!

function ψ_fast_ballistic!(du, u, p, t)
    
    normalize_u!(u, p.n_states)

    update_r!(u, p.r, p.r_idx)

    p.update_params(p, p.r, t)

    update_ψ!(p.ψ, u, p.n_states)

    update_fields_fast!(p, p.r, t)
    
    update_eiωt_new!(p.eiω0ts, p.ω0s, t)

    Heisenberg_turbo_state!(p.ψ, p.eiω0ts, -1)

    update_ψq!(p.ψ_q, p.d_ge, p.d_eg, p.ψ, p.n_g)

    update_d_exp!(p.d_exp, p.ψ, p.ψ_q, p.n_g)

    _ = update_force!(p, p.F, p.d_exp, p.kEs)
     
    update_dψ!(p.dψ, p.ψ_q, p.E_total, p.n_states, p.n_g)

    p.add_terms_dψ(p.dψ, p.ψ, p, p.r, t, 1) # custom terms to add to dψ

    Heisenberg_turbo_state!(p.dψ, p.eiω0ts, +1)

    update_du_ballistic!(du, u, p.dψ, p.ψ, p.n_states, p.n_g, p.r_idx, p.F, p.v_idx, p.F_idx, p.m)

    return nothing
end
export ψ_fast_ballistic!

@inline function normalize_u!(u, n_states)
    u_norm2 = zero(eltype(u))
    @turbo for i ∈ 1:2n_states
        u_norm2 += u[i]^2
    end
    u_norm = sqrt(u_norm2)
    @turbo for i ∈ 1:2n_states
        u[i] /= u_norm
    end
    return nothing
end

@inline function update_r!(u, r, r_idx)
    @inbounds @fastmath for k ∈ 1:3
        r[k] = u[r_idx+k]
    end
    return nothing
end

@inline function update_ψ!(ψ, u, n_states)
    @turbo for i ∈ eachindex(ψ)
        ψ.re[i] = u[i]
        ψ.im[i] = u[i+n_states]
    end
    return nothing
end

@inline function update_u!(u, ψ, n_states)
    @turbo for i ∈ 1:n_states
        u[i] = ψ.re[i]
        u[i+n_states] = ψ.im[i]
    end
    return nothing
end
export update_u!

@inline function update_du!(du, u, dψ, ψ, n_states, n_g, r_idx, F, v_idx, F_idx, m)
    @turbo for i ∈ eachindex(dψ)
        du[i] = dψ.re[i]
        du[i+n_states] = dψ.im[i]
    end
    @inbounds @fastmath for k ∈ 1:3
        du[r_idx+k] = u[v_idx+k]
        du[v_idx+k] = F[k] / m
        u[F_idx+k] = F[k]
        du[F_idx+k+3] = F[k] # integrated force
    end
    @turbo for i ∈ 1:(n_states-n_g) # combine this with loop below?
        # ψ_i_pop = ψ.re[n_g+i]^2 + ψ.im[n_g+i]^2
        ψ_i_pop = u[n_g+i]^2 + u[n_states+n_g+i]^2
        # integrated excited state population
        du[2n_states+i] = ψ_i_pop
    end
    @turbo for i ∈ 1:(n_states-n_g)
        # non-hermitian part of Hamiltonian, -im/2, but multiplied by -im also
        du[n_g+i] -= u[n_g+i]/2
        du[n_states+n_g+i] -= u[n_states+n_g+i]/2
    end
    return nothing
end

@inline function update_du_ballistic!(du, u, dψ, ψ, n_states, n_g, r_idx, F, v_idx, F_idx, m)
    update_du!(du, u, dψ, ψ, n_states, n_g, r_idx, F, v_idx, F_idx, m)
    @inbounds @fastmath for k ∈ 1:3
        du[v_idx+k] = 0.
    end
end

# should this just be put directly into du? probably faster since we use one less function
@inline function update_dψ!(dψ, ψ_q, E, n_states, n_g)
    @turbo for i ∈ 1:n_g
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for q ∈ 1:3
            E_q_re = E.re[q]
            E_q_im = E.im[q]
            ψ_q_re = ψ_q.re[i,q]
            ψ_q_im = ψ_q.im[i,q]
            
            dψ_i_re += ψ_q_re * E_q_re - ψ_q_im * E_q_im
            dψ_i_im += ψ_q_re * E_q_im + ψ_q_im * E_q_re
        end
        dψ.re[i] = dψ_i_im # multiply by -im
        dψ.im[i] = -dψ_i_re
    end
    @turbo for i ∈ n_g+1:n_states
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for q ∈ 1:3
            E_q_re = E.re[q]
            E_q_im = -E.im[q] # conjugate for the excited states
            ψ_q_re = ψ_q.re[i,q]
            ψ_q_im = ψ_q.im[i,q]
            
            dψ_i_re += ψ_q_re * E_q_re - ψ_q_im * E_q_im
            dψ_i_im += ψ_q_re * E_q_im + ψ_q_im * E_q_re
        end
        dψ.re[i] = dψ_i_im # multiply by -im
        dψ.im[i] = -dψ_i_re
    end
    return nothing
end
export update_dψ!

"""
    Evalute ψ_q ≡ d_q ψ.

    Break up psi into ground and excited? Also do we really need the excited states, or can we do everything with ground states and taking conjugates?
"""
@inline function update_ψq!(ψ_q, d_ge, d_eg, ψ, n_g)
    @turbo for q ∈ 1:3
        for i ∈ axes(d_ge,1)
            ψq_re_i = zero(eltype(ψ_q.re))
            ψq_im_i = zero(eltype(ψ_q.im))
            for j ∈ axes(d_ge,2)
                d_q_ij = d_ge[i,j,q]
                ψ_re_j = ψ.re[n_g+j]
                ψ_im_j = ψ.im[n_g+j]
                ψq_re_i += d_q_ij * ψ_re_j
                ψq_im_i += d_q_ij * ψ_im_j
            end
            ψ_q.re[i,q] = ψq_re_i
            ψ_q.im[i,q] = ψq_im_i
        end
    end
    @turbo for q ∈ 1:3
        for i ∈ axes(d_eg,1)
            ψq_re_i = zero(eltype(ψ_q.re))
            ψq_im_i = zero(eltype(ψ_q.im))
            for j ∈ axes(d_eg,2)
                d_q_ij = d_eg[i,j,q]
                ψ_re_j = ψ.re[j]
                ψ_im_j = ψ.im[j]
                ψq_re_i += d_q_ij * ψ_re_j
                ψq_im_i += d_q_ij * ψ_im_j
            end
            ψ_q.re[n_g+i,q] = ψq_re_i
            ψ_q.im[n_g+i,q] = ψq_im_i
        end
    end
    return nothing
end

# can maybe make it so that we don't have to take the sincos for all states if there are degeneracies
@inline function update_eiωt_new!(eiω0ts, ω0s, t)
    @turbo for i ∈ eachindex(eiω0ts)
        eiω0ts.im[i], eiω0ts.re[i] = sincos(ω0s[i] * t)
    end
    return nothing
end
export update_eiωt_new!

"""
    Note that this isn't actually d_exp, since we only go over the ground states.
    So this is the d^+ part of d = d^+ + d^-
"""
@inline function update_d_exp!(d_exp, ψ, ψ_q, n_g)
    @turbo for q ∈ 1:3
        re = zero(eltype(ψ.re))
        im = zero(eltype(ψ.im))
        for i ∈ 1:n_g
            ψ_re = ψ.re[i] # take conjugate
            ψ_im = -ψ.im[i]
            ψq_re = ψ_q.re[i,q]
            ψq_im = ψ_q.im[i,q]
            re += ψ_re * ψq_re - ψ_im * ψq_im
            im += ψ_re * ψq_im + ψq_re * ψ_im
        end
        d_exp.re[q] = re
        d_exp.im[q] = im
    end
    return nothing
end

@inline function update_force!(p, F, d_exp, kEs)
    if p.sim_params.trap_scalar !=0
        gaussian_trap_scalar, ∇H = H_func_blue_red(p)
        f_ODT = -H_expectation(p.ψ,p.sim_params.H_ODT_matrix,p.sim_params.ODT_idxs,p.sim_params.trap_scalar).*∇H
    else
        f_ODT = [0.,0.,0.]
        gaussian_trap_scalar = 0.
    end
    @turbo for k ∈ 1:3
        F_k = zero(eltype(F))
        for q ∈ 1:3
            d_q_re = d_exp.re[q]
            d_q_im = d_exp.im[q]
            
            # multiply by -im
            E_kq_im = -(kEs.re[k,q] - kEs.re[k+3,q])
            E_kq_re = (kEs.im[k,q] - kEs.im[k+3,q])
            
            F_k_a_re = d_q_re * E_kq_re - d_q_im * E_kq_im

            F_k -= 2F_k_a_re

        end
        F[k] = F_k + f_ODT[k]
    end
    return gaussian_trap_scalar
end
export update_force!

function stochastic_collapse_new!(integrator)

    u = integrator.u
    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_g
    d_ge = p.d_ge
    ψ = p.ψ
    
    p⁺ = zero(eltype(ψ.re))
    p⁰ = zero(eltype(ψ.re))
    p⁻ = zero(eltype(ψ.re))

    @turbo for i ∈ 1:n_excited
        c_i_re = ψ.re[n_ground + i] 
        c_i_im = -ψ.im[n_ground + i] # take conjugate
        for j ∈ 1:n_excited
            c_j_re = ψ.re[n_ground + j]
            c_j_im = ψ.im[n_ground + j]
            re = c_i_re * c_j_re - c_i_im * c_j_im
            for k ∈ 1:n_ground
                p⁺ += re * d_ge[k,i,1] * d_ge[k,j,1] # assume that d is real
                p⁰ += re * d_ge[k,i,2] * d_ge[k,j,2]
                p⁻ += re * d_ge[k,i,3] * d_ge[k,j,3]
            end
            # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
            # whereas the polarization of the emitted photon is m_g - m_e
        end
    end

    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end

    # zero ground state amplitudes
    @turbo for i ∈ 1:n_ground
        # ψ.re[i] = zero(eltype(ψ.re))
        # ψ.im[i] = zero(eltype(ψ.im))
        u[i] = zero(eltype(u))
        u[i+n_states] = zero(eltype(u))
    end

    # decay from excited to ground state
    @turbo for i ∈ 1:n_ground
        for j ∈ 1:n_excited
            d = d_ge[i,j,pol]
            # ψ.re[i] += d * ψ.re[n_ground+j]
            # ψ.im[i] += d * ψ.im[n_ground+j]
            u[i] += d * ψ.re[n_ground+j]
            u[i+n_states] += d * ψ.im[n_ground+j]
        end
    end
    
    # zero excited state amplitudes
    @turbo for i ∈ 1:n_excited
        # ψ.re[n_ground+i] = zero(eltype(ψ.re))
        # ψ.im[n_ground+i] = zero(eltype(ψ.im))
        u[n_ground+i] = zero(eltype(u))
        u[i+n_states+n_ground] = zero(eltype(u))
    end

    # zero integrated excited state populations - # add this with loop above???
    @turbo for i ∈ 1:n_excited
        u[2n_states+i] = zero(eltype(u))
    end

    # add diffusion
    time_before_decay = integrator.t - p.last_decay_time
    @inbounds @fastmath for i ∈ 1:3
        kick = sqrt( 2p.diffusion_constant[i] * time_before_decay ) / p.m
        u[2n_states + n_excited + 3 + i] += rand((-1,1)) * kick
        # u[2n_states + n_excited + 3 + i] += rand(Normal(0,kick))
    end
    p.last_decay_time = integrator.t

    # add spontaneous decay
    if p.add_spontaneous_decay_kick
        @inbounds @fastmath for i ∈ 1:3
            u[2n_states + n_excited + 3 + i] += rand((-1,1)) / p.m
        end
    end

    p.time_to_decay = rand(p.decay_dist)
    p.n_scatters += 1

    return nothing
end
export stochastic_collapse_new!

@inline function condition_new(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = zero(eltype(u))
    @inbounds @fastmath for i ∈ 1:(p.n_states-p.n_g)
        p_i = u[2p.n_states+i]
        integrated_excited_pop += p_i
    end
    _condition = integrated_excited_pop - p.time_to_decay
    return _condition
end
export condition_new

@inline function condition_discrete(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = zero(eltype(u))
    @inbounds @fastmath for i ∈ 1:(p.n_states-p.n_g)
        p_i = u[2p.n_states+i]
        integrated_excited_pop += p_i
    end
    _condition = integrated_excited_pop - p.time_to_decay
    return _condition > 0
end
export condition_discrete

function stochastic_collapse_no_diffusion!(integrator)

    u = integrator.u
    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_g
    d_ge = p.d_ge
    ψ = p.ψ
    
    p⁺ = zero(eltype(ψ.re))
    p⁰ = zero(eltype(ψ.re))
    p⁻ = zero(eltype(ψ.re))

    @turbo for i ∈ 1:n_excited
        c_i_re = ψ.re[n_ground + i] 
        c_i_im = -ψ.im[n_ground + i] # take conjugate
        for j ∈ 1:n_excited
            c_j_re = ψ.re[n_ground + j]
            c_j_im = ψ.im[n_ground + j]
            re = c_i_re * c_j_re - c_i_im * c_j_im
            for k ∈ 1:n_ground
                p⁺ += re * d_ge[k,i,1] * d_ge[k,j,1] # assume that d is real
                p⁰ += re * d_ge[k,i,2] * d_ge[k,j,2]
                p⁻ += re * d_ge[k,i,3] * d_ge[k,j,3]
            end
            # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
            # whereas the polarization of the emitted photon is m_g - m_e
        end
    end

    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end

    # zero ground state amplitudes
    @turbo for i ∈ 1:n_ground
        # ψ.re[i] = zero(eltype(ψ.re))
        # ψ.im[i] = zero(eltype(ψ.im))
        u[i] = zero(eltype(u))
        u[i+n_states] = zero(eltype(u))
    end

    # decay from excited to ground state
    @turbo for i ∈ 1:n_ground
        for j ∈ 1:n_excited
            d = d_ge[i,j,pol]
            # ψ.re[i] += d * ψ.re[n_ground+j]
            # ψ.im[i] += d * ψ.im[n_ground+j]
            u[i] += d * ψ.re[n_ground+j]
            u[i+n_states] += d * ψ.im[n_ground+j]
        end
    end
    
    # zero excited state amplitudes
    @turbo for i ∈ 1:n_excited
        # ψ.re[n_ground+i] = zero(eltype(ψ.re))
        # ψ.im[n_ground+i] = zero(eltype(ψ.im))
        u[n_ground+i] = zero(eltype(u))
        u[i+n_states+n_ground] = zero(eltype(u))
    end

    # zero integrated excited state populations - # add this with loop above???
    @turbo for i ∈ 1:n_excited
        u[2n_states+i] = zero(eltype(u))
    end

    p.time_to_decay = rand(p.decay_dist)
    p.n_scatters += 1

    return nothing
end
export stochastic_collapse_no_diffusion!



"""
    Compute kEs for a subset of laser frequencies (f_start:f_end).
"""
@inline function update_kEs_for_group!(kEs_out, as, ϵs, f_start, f_end)
    @inbounds @fastmath for q ∈ 1:3
        for k ∈ 1:6
            E_kq_re = zero(eltype(kEs_out.re))
            E_kq_im = zero(eltype(kEs_out.im))
            for f ∈ f_start:f_end
                a_re = as.re[k,f]
                a_im = as.im[k,f]
                ϵ_re = ϵs.re[k,f,q]
                ϵ_im = -ϵs.im[k,f,q]
                E_kq_re += ϵ_re * a_re - ϵ_im * a_im
                E_kq_im += ϵ_re * a_im + ϵ_im * a_re
            end
            kEs_out.re[k,q] = E_kq_re
            kEs_out.im[k,q] = E_kq_im
        end
    end
    return nothing
end
export update_kEs_for_group!

"""
    Compute dipole expectation value for a subset of ground states (g_start:g_end).
"""
@inline function update_d_exp_for_group!(d_exp_out, ψ, ψ_q, g_start, g_end)
    @inbounds @fastmath for q ∈ 1:3
        re = zero(eltype(ψ.re))
        im = zero(eltype(ψ.im))
        for i ∈ g_start:g_end
            ψ_re = ψ.re[i]
            ψ_im = -ψ.im[i]
            ψq_re = ψ_q.re[i,q]
            ψq_im = ψ_q.im[i,q]
            re += ψ_re * ψq_re - ψ_im * ψq_im
            im += ψ_re * ψq_im + ψq_re * ψ_im
        end
        d_exp_out.re[q] = re
        d_exp_out.im[q] = im
    end
    return nothing
end
export update_d_exp_for_group!

"""
    Compute dψ with per-group E fields.
"""
@inline function update_dψ_grouped!(dψ, ψ_q, ψ, d_eg, E_g1, E_g2, n_states, n_g, g1_g_start, g1_g_end, g2_g_start, g2_g_end)
    # Ground states group 1: driven by E_g1
    @inbounds @fastmath for i ∈ g1_g_start:g1_g_end
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for q ∈ 1:3
            E_q_re = E_g1.re[q]
            E_q_im = E_g1.im[q]
            ψ_q_re = ψ_q.re[i,q]
            ψ_q_im = ψ_q.im[i,q]
            dψ_i_re += ψ_q_re * E_q_re - ψ_q_im * E_q_im
            dψ_i_im += ψ_q_re * E_q_im + ψ_q_im * E_q_re
        end
        dψ.re[i] = dψ_i_im
        dψ.im[i] = -dψ_i_re
    end

    # Ground states group 2: driven by E_g2
    @inbounds @fastmath for i ∈ g2_g_start:g2_g_end
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for q ∈ 1:3
            E_q_re = E_g2.re[q]
            E_q_im = E_g2.im[q]
            ψ_q_re = ψ_q.re[i,q]
            ψ_q_im = ψ_q.im[i,q]
            dψ_i_re += ψ_q_re * E_q_re - ψ_q_im * E_q_im
            dψ_i_im += ψ_q_re * E_q_im + ψ_q_im * E_q_re
        end
        dψ.re[i] = dψ_i_im
        dψ.im[i] = -dψ_i_re
    end

    # Excited states
    n_excited = n_states - n_g
    @inbounds @fastmath for i ∈ 1:n_excited
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))

        # Group 1 contribution
        for q ∈ 1:3
            E_q_re = E_g1.re[q]
            E_q_im = -E_g1.im[q]
            ψq_re = zero(eltype(ψ.re))
            ψq_im = zero(eltype(ψ.im))
            for g ∈ g1_g_start:g1_g_end
                d_val = d_eg[i,g,q]
                ψq_re += d_val * ψ.re[g]
                ψq_im += d_val * ψ.im[g]
            end
            dψ_i_re += ψq_re * E_q_re - ψq_im * E_q_im
            dψ_i_im += ψq_re * E_q_im + ψq_im * E_q_re
        end

        # Group 2 contribution
        for q ∈ 1:3
            E_q_re = E_g2.re[q]
            E_q_im = -E_g2.im[q]
            ψq_re = zero(eltype(ψ.re))
            ψq_im = zero(eltype(ψ.im))
            for g ∈ g2_g_start:g2_g_end
                d_val = d_eg[i,g,q]
                ψq_re += d_val * ψ.re[g]
                ψq_im += d_val * ψ.im[g]
            end
            dψ_i_re += ψq_re * E_q_re - ψq_im * E_q_im
            dψ_i_im += ψq_re * E_q_im + ψq_im * E_q_re
        end

        dψ.re[n_g+i] = dψ_i_im
        dψ.im[n_g+i] = -dψ_i_re
    end
    return nothing
end
export update_dψ_grouped!

"""
    Compute force using per-group d_exp and kEs, avoiding cross-manifold oscillations.
"""
@inline function update_force_grouped!(p, F, d_exp_g1, d_exp_g2, kEs_g1, kEs_g2)
    if p.sim_params.trap_scalar != 0
        gaussian_trap_scalar, ∇H = H_func_blue_red(p)
        f_ODT = -H_expectation(p.ψ, p.sim_params.H_ODT_matrix, p.sim_params.ODT_idxs, p.sim_params.trap_scalar) .* ∇H
    else
        f_ODT = (zero(eltype(F)), zero(eltype(F)), zero(eltype(F)))
        gaussian_trap_scalar = zero(eltype(F))
    end
    @inbounds @fastmath for k ∈ 1:3
        F_k = zero(eltype(F))
        for q ∈ 1:3
            # Group 1
            d_q_re = d_exp_g1.re[q]
            d_q_im = d_exp_g1.im[q]
            E_kq_im = -(kEs_g1.re[k,q] - kEs_g1.re[k+3,q])
            E_kq_re = (kEs_g1.im[k,q] - kEs_g1.im[k+3,q])
            F_k -= 2 * (d_q_re * E_kq_re - d_q_im * E_kq_im)

            # Group 2
            d_q_re = d_exp_g2.re[q]
            d_q_im = d_exp_g2.im[q]
            E_kq_im = -(kEs_g2.re[k,q] - kEs_g2.re[k+3,q])
            E_kq_re = (kEs_g2.im[k,q] - kEs_g2.im[k+3,q])
            F_k -= 2 * (d_q_re * E_kq_re - d_q_im * E_kq_im)
        end
        F[k] = F_k + f_ODT[k]
    end
    return gaussian_trap_scalar
end
export update_force_grouped!

"""
    RWA-grouped version of ψ_fast!.

    Computes per-group E fields and applies each only to its target ground states,
    eliminating stiffness from cross-manifold couplings (e.g. v=0 ↔ v=1 at 17.5 THz).
"""
function ψ_fast_grouped!(du, u, p, t)

    normalize_u!(u, p.n_states)

    update_r!(u, p.r, p.r_idx)

    p.update_params(p, p.r, t)

    update_ψ!(p.ψ, u, p.n_states)

    # Field amplitudes
    update_rs!(p.rs, p.r, p.denom, p.iris_factor)
    update_ϕs!(p.ϕs, p.ωs, p.r, t)
    update_as!(p.ϕs, p.as, p.rs, p.sats)

    # Per-group kEs and E_total
    update_kEs_for_group!(p.kEs_g1, p.as, p.ϵs, p.g1_f_start, p.g1_f_end)
    update_kEs_for_group!(p.kEs_g2, p.as, p.ϵs, p.g2_f_start, p.g2_f_end)
    update_E_total!(p.E_g1, p.kEs_g1)
    update_E_total!(p.E_g2, p.kEs_g2)

    # Interaction picture transform
    update_eiωt_new!(p.eiω0ts, p.ω0s, t)

    Heisenberg_turbo_state!(p.ψ, p.eiω0ts, -1)

    # ψ_q = d · ψ
    update_ψq!(p.ψ_q, p.d_ge, p.d_eg, p.ψ, p.n_g)

    # Per-group d_exp and force
    update_d_exp_for_group!(p.d_exp_g1, p.ψ, p.ψ_q, p.g1_g_start, p.g1_g_end)
    update_d_exp_for_group!(p.d_exp_g2, p.ψ, p.ψ_q, p.g2_g_start, p.g2_g_end)

    gaussian_trap_scalar = update_force_grouped!(p, p.F, p.d_exp_g1, p.d_exp_g2, p.kEs_g1, p.kEs_g2)

    # Per-group dψ
    update_dψ_grouped!(p.dψ, p.ψ_q, p.ψ, p.d_eg, p.E_g1, p.E_g2, p.n_states, p.n_g, p.g1_g_start, p.g1_g_end, p.g2_g_start, p.g2_g_end)

    p.add_terms_dψ(p.dψ, p.ψ, p, p.r, t, gaussian_trap_scalar)

    Heisenberg_turbo_state!(p.dψ, p.eiω0ts, +1)

    update_du!(du, u, p.dψ, p.ψ, p.n_states, p.n_g, p.r_idx, p.F, p.v_idx, p.F_idx, p.m)

    return nothing
end
export ψ_fast_grouped!

function ψ_fast_ballistic_grouped!(du, u, p, t)

    normalize_u!(u, p.n_states)

    update_r!(u, p.r, p.r_idx)

    p.update_params(p, p.r, t)

    update_ψ!(p.ψ, u, p.n_states)

    update_rs!(p.rs, p.r, p.denom, p.iris_factor)
    update_ϕs!(p.ϕs, p.ωs, p.r, t)
    update_as!(p.ϕs, p.as, p.rs, p.sats)

    update_kEs_for_group!(p.kEs_g1, p.as, p.ϵs, p.g1_f_start, p.g1_f_end)
    update_kEs_for_group!(p.kEs_g2, p.as, p.ϵs, p.g2_f_start, p.g2_f_end)
    update_E_total!(p.E_g1, p.kEs_g1)
    update_E_total!(p.E_g2, p.kEs_g2)

    update_eiωt_new!(p.eiω0ts, p.ω0s, t)

    Heisenberg_turbo_state!(p.ψ, p.eiω0ts, -1)

    update_ψq!(p.ψ_q, p.d_ge, p.d_eg, p.ψ, p.n_g)

    update_d_exp_for_group!(p.d_exp_g1, p.ψ, p.ψ_q, p.g1_g_start, p.g1_g_end)
    update_d_exp_for_group!(p.d_exp_g2, p.ψ, p.ψ_q, p.g2_g_start, p.g2_g_end)

    _ = update_force_grouped!(p, p.F, p.d_exp_g1, p.d_exp_g2, p.kEs_g1, p.kEs_g2)

    update_dψ_grouped!(p.dψ, p.ψ_q, p.ψ, p.d_eg, p.E_g1, p.E_g2, p.n_states, p.n_g, p.g1_g_start, p.g1_g_end, p.g2_g_start, p.g2_g_end)

    p.add_terms_dψ(p.dψ, p.ψ, p, p.r, t, 1)

    Heisenberg_turbo_state!(p.dψ, p.eiω0ts, +1)

    update_du_ballistic!(du, u, p.dψ, p.ψ, p.n_states, p.n_g, p.r_idx, p.F, p.v_idx, p.F_idx, p.m)

    return nothing
end
export ψ_fast_ballistic_grouped!