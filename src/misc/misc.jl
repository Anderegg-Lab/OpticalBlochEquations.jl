"""
    Flip the k-vector corresponding to the polarization.
"""
function flip(ϵ)
    return SVector{3,ComplexF64}(ϵ[3],-ϵ[2],ϵ[1])
end

"""
    Return the integer type corresponding to a float type.
"""
function get_int_type(T)
    if T == Float32
        return Int32
    elseif T == Float64
        return Int64
    end
end

"""
    Find the number of ground states based on the transition dipole array `d`.
"""
function find_n_g(d)
    n_g = 0
    for j ∈ axes(d,2)
        set_n_g = true
        for i ∈ axes(d,1)
            for q ∈ 1:3
                if norm(d[i,j,q]) > 0 && (i < j)
                    set_n_g = false
                end
            end
        end
        if set_n_g
            n_g = j
        end
    end
    return n_g
end