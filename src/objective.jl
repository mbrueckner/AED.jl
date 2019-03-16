function gsd_utility(A::Vector{Int}, u::Matrix{T}, pd::Array{T,3}, n, param; tau=0.0, h=0.001) where T <: Real
    obj = 0.0
    B, J, K = size(pd)

    for b in 1:B
        cost = n[b,end]
        reject = false
        for j in 1:J
            cost = n[b,j]
            for k in A
                if pd[b,j,k] > u[j,k]
                    reject = true
                    break
                end
            end
            if reject
                break
            end
        end
        obj -= cost
    end

    obj / B
end

function sponsor_utility(A::Vector{Int}, u::Matrix{T}, pd::Array{T,3}, n, param::Matrix{T}; tau=0.0, h=0.001) where T <: Real
    obj = 0.0
    B, J, K = size(pd)

    for b in 1:B
        pp = [1 - param[b,end], param[b,end]]
        cost = h*n[b,end]
        reject = false
        for j in 1:J
            cost = h*n[b,j]
            for k in A
                if pd[b,j,k] > u[j,k]
                    obj += pp[k] ## reward for reject
                    reject = true
                end
            end
            if reject
                break
            end
        end
        obj -= cost
    end

    obj / B
end

function public_utility(A::Vector{Int}, u::Matrix{T}, pd::Array{T,3}, n, param::Matrix{T}; tau=0.0, h=0.001) where T <: Real
    obj = 0.0
    B, J, K = size(pd)

    for b in 1:B
        pp = [1 - param[b,end], param[b,end]]
        cost = h*n[b,end]
        reject = false
        for j in 1:J
            cost = h*n[b,j]
            for k in A
                if pd[b,j,k] > u[j,k]
                    if param[b,k] < tau
                        obj += pp[k] ## reward for TP
                    end
                    reject = true
                end
            end

            if reject
                break
            end
        end

        ## reward for TN selection
        for k in 1:K
            if (k ∉ A) & (param[b,k] >= tau)
                obj += pp[k]
            end
        end

        obj -= cost
    end

    obj / B
end

function alpha(pd::Array{T,3}, upper::Matrix{T}; A=1:size(pd,3)) where T <: Real
    B, J, K = size(pd)
    reject = 0

    @inbounds for b in 1:B
        stop = false
        @inbounds for j in 1:J
            @inbounds for k in A
                if pd[b,j,k] > upper[j,k]
                    reject += 1
                    stop = true
                    break
                end
            end
            if stop
                break
            end
        end
    end

    reject / B
end

function objective(A::Vector{Int}, x::Matrix, type1, type2, fsim, fsim0, fsim1; h=0.00025, utility=gsd_utility)
    ## actual type I and type II error for intersection hypothesis
    real_t1 = alpha(fsim0.post_pd, x; A=A)
    real_t2 = 1-alpha(fsim1.post_pd, x; A=A)

    if (real_t1 <= type1) & (real_t2 <= type2)
        utility(A, x, fsim.post_pd, fsim.n, fsim.param; h=h), real_t1, real_t2
    else
        -Inf, real_t1, real_t2
    end
end
