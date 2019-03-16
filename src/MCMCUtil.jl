module MCMCUtil

using LinearAlgebra
using Distributions

updateMean(t::Int, Xt::Float64, m0::Float64) = (Xt + m0*(t-1)) / t
updateMean(t::Int, Xt::Array{Float64,1}, m0::Array{Float64,1}) = @. (Xt + m0*(t-1)) / t

function updateVar(t::Int, Xt::Float64, mt::Float64, m0::Float64, V0::Float64)
    return V0*(t-2)/(t-1) + t*(mt-m0)^2
end

function updateVar(t::Int, Xt::Array{Float64,1}, mt::Array{Float64,1},
                   m0::Array{Float64,1}, V0::Array{Float64,2}, eps::Matrix{Float64})
    return V0.*(t-2)/(t-1) + 1/(t-1).*((t-1).*(m0 * transpose(m0)) .- t.* (mt * transpose(mt)) .+ (Xt * transpose(Xt)) + eps)
end

function MH_single(target_log::Function, init, prop)
    r = target_log(prop) - target_log(init)

    if log(rand()) < r
        return prop, 1
    else
        return init, 0
    end
end

function MH_single(target_log::Function, init::T, proposal::Function, init_loglik::T) where T <: Real
    prop = proposal(init)
    tlp = target_log(prop)
    tli = target_log(init, init_loglik)

    if log(rand()) < (tlp[1] - tli[1])
        return prop, 1, tlp[2]
    else
        return init, 0, tlp[2]
    end
end

## single adaptive Metropolis step
function AM_RR_single(target_log::Function, init::Vector{T}, V::Matrix{T},
                      t::Int, t0::Int=2*length(init),
                      s::T=(2.38)^2/length(init), beta::T=0.05) where T <: Real

    d = length(init)
    proposal = zeros(T, d)

    if (t <= t0) || (rand() < beta)
        proposal = rand(MvNormal(init, (0.1)^2*diagm(0 => ones(eltype(init), d))/d))
    else
        proposal = rand(MvNormal(init, s*V))
    end

    if log(rand()) < (target_log(proposal) - target_log(init))
        return proposal, 1
    else
        return init, 0
    end
end

function AM_RR_single(target_log::Function, init::Float64, V::Float64,
                      t::Int, t0::Int=2, s::Float64=2.38, beta::Float64=0.05)

    proposal::Float64 = 0

    if (t <= t0) | (rand(1)[1] < beta)
        proposal = rand(Normal(init, 0.1))
    else
        if(s*V == 0)
            proposal = init
        else
            proposal = rand(Normal(init, s*sqrt(V)))
        end
    end

    r = target_log(proposal) - target_log(init)

    if log(rand()) < r ##min(1.0, r[1])
        return proposal, 1
    else
        return init, 0
    end
end

end
