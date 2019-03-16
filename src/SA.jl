module SA

##export simulated_annealing, random_search, grid_search

using Distributions
using SharedArrays

function grid_search(M, lower, upper, objective::Function)
    max_obj = objective(upper)
    best = lower
    K = length(M)

    ranges = map(i -> lower[i]:(upper[i]-lower[i])/M[i]:upper[i], 1:K)

    @inbounds for el in Base.product(ranges...)
        new_obj = objective(el)
        if new_obj > max_obj
            best, max_obj = el, new_obj
        end
    end

    best, max_obj
end

function simulated_annealing(M::Int, init::T,
                             kernel::Function,
                             objective::Function,
                             temperature::Function = t -> 1/log(t)) where T
    x = Vector{T}(undef,M+1)
    x[1] = init

    old_obj = objective(x[1])
    max_obj = old_obj
    accept = 0
    best_x = x[1]

    obj = Vector{T}(undef,M+1)
    obj[1] = old_obj

    @inbounds for t in 1:M
        proposal = kernel(x[t], temperature(t+1))
        new_obj = objective(proposal)

        if log(rand()) <= (new_obj - old_obj) / temperature(t+1)
            x[t+1] = proposal
            old_obj = new_obj

            if max_obj < new_obj
                max_obj = new_obj
                best_x = proposal
                accept += 1
            end
        else
            x[t+1] = x[t]
        end

        obj[t+1] = old_obj
    end

    max_obj, best_x, accept/(M+1)
    ##max_obj, best_x, obj[end], x[end], accept/(M+1)
end

function random_search(M::Int, init::T, kernel::Function, objective::Function) where T
    max_obj = objective(init)
    x::T = init
    accept = 0

    for t in 1:M
        proposal = kernel(x)
        new_obj = objective(proposal)

        if new_obj > max_obj
            x = proposal
            max_obj = new_obj
            accept += 1
        end
    end

    max_obj, x, accept/(M+1)
end

function test_kernel(x, temp)
    rand(Normal(0, 1), 2)
end

function test_kernel2(x, temp)
    rand(Uniform(-4,4), 2)
end

function test_obj(x)
    pdf(MultivariateNormal([1.0, -0.5], eye(2)), x)
end

function rosenbrock(x)
    -((1 - x[1])^2 + 100*(x[2] - x[1]^2)^2)
end

function himmelblau(x)
  -((x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2)
end

end
