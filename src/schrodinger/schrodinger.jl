function initialize_prob(
    ψ₀,
    states,
    fields,
    d,
    sim_params,
    update_p,
    add_terms_dψ
    )

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    fields = StructArray(fields)

    for i ∈ eachindex(states)
        states.E[i] *= 2π
    end
    for i ∈ eachindex(fields)
        fields.ω[i] *= 2π
    end

    type_complex = ComplexF64

    H = StructArray( zeros(type_complex, n_states, n_states) )

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]
    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if (abs(dme) > 1e-10) & (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]

    ψ = deepcopy(ψ₀)
    dψ = deepcopy(ψ)
    ψ_soa = StructArray(ψ)
    dψ_soa = StructArray(dψ)

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    r = SVector(0.,0.,0.)

    p = MutableNamedTuple(
        r=r,
        H=H,
        ψ=ψ, 
        dψ=dψ, 
        ψ_soa=ψ_soa, 
        dψ_soa=dψ_soa, 
        ω=ω, 
        eiωt=eiωt,
        states=states, 
        fields=fields, 
        d=d,
        E=E, 
        E_k=E_k,
        ds=ds, 
        ds_state1=ds_state1, 
        ds_state2=ds_state2,
        sim_params=sim_params,
        update_p=update_p,
        add_terms_dψ=add_terms_dψ
        )
end
export initialize_prob

function schrodinger!(dψ, ψ, p, t)

    @unpack ψ_soa, dψ_soa, r, ω, fields, H, E_k, ds, ds_state1, ds_state2, eiωt, states = p

    p.update_p(p, r, t)

    base_to_soa!(ψ, ψ_soa)
    
    update_H_schrodinger!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2)
    
    update_eiωt!(eiωt, ω, t)
    Heisenberg_turbo_state!(ψ_soa, eiωt, -1)

    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    p.add_terms_dψ(dψ_soa, ψ_soa, p, r, t)

    Heisenberg_turbo_state!(dψ_soa, eiωt, +1)
    soa_to_base!(dψ, dψ_soa)

    return nothing
end
export schrodinger!