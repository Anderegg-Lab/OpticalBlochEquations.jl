"""
    param_scan(evaluate_func, params, scan_params; nthreads)

    Evaluates an `ODEProblem` by looping over both the set of params [`outer_scan_params`; `inner_scan_params`].

    update_func(params, outer_scan_param) --> updates the parameters `p` of the passed `prob`
    evaluate_func() --> 
"""
function param_scan(update_func::F1, evaluate_func::F2, prob::ODEProblem, outer_scan_params, inner_scan_params; nthreads=Threads.nthreads()) where {F1,F2}

    # Make an iterator from `inner_scan_params`, break them into chunks to prepare for threading
    iterator_product = Iterators.product(values(inner_scan_params)...)
    n_chunks = cld(length(iterator_product), nthreads)
    iterator = Iterators.partition(iterator_product, n_chunks)
    iterated_values = keys(inner_scan_params)
    n_distinct = sum(length.(iterator))

    params_chunks = [deepcopy(params) for _ ∈ 1:nthreads]
    
    prog_bar = Progress(n_distinct)

    tasks = Vector{Task}(undef, nthreads)
    return_values = []

    for outer_scan_param ∈ outer_scan_params
        update_func(prob, outer_scan_param)

        @sync for (i, scan_params_chunk) ∈ enumerate(iterator)

            tasks[i] = Threads.@spawn begin
                iterated_values = keys(scan_params)

                params_chunk = params_chunks[i]
                forces_chunk = SVector{3, Float64}[]

                t_end = 300
                tspan = (0.0, t_end)
                times = range(t_end - params_chunk.period, t_end, 1000)

                ρ0 = zeros(ComplexF64, (length(params_chunk.states), length(params_chunk.states)))
                ρ0[1,1] = 1.0
                prob = ODEProblem(ρ!, ρ0, tspan, params_chunk)

                for j ∈ eachindex(scan_params_chunk)
                    for (k, iterated_value) ∈ enumerate(iterated_values)
                        setproperty!(params_chunk, iterated_value, scan_params_chunk[j][k])
                    end
                    round_params(params_chunk) # round params to `freq_res` accuracy just in case they were updated
                    sol = solve(prob, alg=DP5(), abstol=1e-5)
                    push!(forces_chunk, evaluate_func(params_chunk, sol))
                    next!(prog_bar)
                end
                forces_chunk
            end
        end
    end
    forces = vcat(fetch.(tasks)...)
    return (iterated_values, iterator_product), forces

end
export force_scan

# Define callbacks to calculate force and terminate integration once it has converged
# condition(u, t, integrator) = false #integrator.p.force
# affect!(integrator) = terminate!(integrator)
# cb = DiscreteCallback(condition, affect!)

function force_callback!(integrator)
    # if integrator.t > 10integrator.p.period
    p = integrator.p
    force = force_noupdate(p) / p.n_force_values
    modded_idx = mod1(p.force_idx, p.n_force_values)
    p.forces[modded_idx] += force
    for i ∈ 1:10
        modded_idx1 = mod1(p.force_idx - i * 100, p.n_force_values)
        modded_idx2 = mod1(modded_idx1 + 1, p.n_force_values)
        p.force_chunks[i] += p.forces[modded_idx1]
        p.force_chunks[i] -= p.forces[modded_idx2]
    end
    p.force_idx += 1
    # end
    return nothing
end
;