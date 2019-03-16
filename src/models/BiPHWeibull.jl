## Multi-arm Cox PH model with shared control group

struct BiPHWeibullData <: AbstractData
    id::Vector{Int}
    entry::Vector{Float64}
    time::Vector{Float64}
    status::Vector{Bool}
    b::Vector{Int}
    z::Vector{Float64}
end

nevents(data::BiPHWeibullData) = sum(data.status)

BiPHWeibullData() = Data(Vector{Int}(undef,0), Vector{Float64}(undef,0), Vector{Float64}(undef,0),
              Vector{Bool}(undef,0), Vector{Int}(undef,0), Vector{Float64}(undef,0))

mutable struct BiPHWeibullParam <: AbstractParam
    shape::NTuple{2,Float64}
    scale::NTuple{2,Float64}
    beta::NTuple{2,Float64}
    prev::Float64 ## biomarker-positive subgroup prevalence
end

struct BiPHWeibullFixedParam <: AbstractFixedParam
    n::Int
    tmax::Float64
    max_entry::Float64
end

struct BiPHWeibullPrior <: AbstractPrior
    shape::NTuple{2,UnivariateDistribution}
    scale::NTuple{2,UnivariateDistribution}
    beta::MultivariateDistribution
    prev::UnivariateDistribution
end

BiPHWeibullPrior() = BiPHWeibullPrior((Gamma(1,1), Gamma(1,1)), (Gamma(1,1), Gamma(1,1)), MvNormal(10*diagm(0 => ones(Float64, 2))), Uniform(0.1, 0.9))

BiPHWeibullModel() = Model(BiPHWeibullParam((1.0, 1.0), (log(2)/5, log(2)/5), (log(0.8), log(0.6)), 1/3), BiPHWeibullFixedParam(400, 36.0, 48.0), generate_data)

BiPHWeibullModel0() = Model(BiPHWeibullParam((1.0, 1.0), (log(2)/5, log(2)/5), (0.0, 0.0), 1/3), BiPHWeibullFixedParam(400, 36.0, 48.0), generate_data)

BiPHWeibullModel(param, fparam, gen_data=generate_data) = Model(param, fparam, gen_data)

struct BiPHWeibullFit <: AbstractFit
    shape::Matrix{Float64}
    scale::Matrix{Float64}
    beta::Matrix{Float64} ## hazard ratios
    prev::Vector{Float64} ## biomarker-positive subgroup prevalence
    accept::Float64 ## acceptance rates
    prior::BiPHWeibullPrior
end

function slice(i::Int, x::BiPHWeibullFit)
    if size(x.beta, 1) == 0
        sample_prior(1, x.prior)[1]
    else
        BiPHWeibullParam((x.shape[i,1], x.shape[i,2]), (x.scale[i,1], x.scale[i,2]), (x.beta[i,1], x.beta[i,2]), x.prev[i])
    end
 end

function sample_prior(B::Int, prior::Prior)
     res = Vector{BiPHWeibullParam}(undef, B)
     for b in 1:B
         res[b] = BiPHWeibullParam((rand(prior.shape[1]), rand(prior.shape[2])),
                                    (rand(prior.scale[1]), rand(prior.scale[2])),
                                    rand(prior.beta), rand(prior.prev))
     end
     res
end

## return prior distribution (before sampling) or posterior sample (after sampling)
## of treatment effect parameter
function get_outcome(x::BiPHWeibullFit)
    if size(x.beta, 1) == 0
        x.prior.beta
    else
        x.beta
    end
end

generate_data(par, fpar) = predict_data(par, fpar, BiPHWeibullData(), 0.0)

function predict_data(par::BiPHWeibullParam, fpar::BiPHWeibullFixedParam, data0::BiPHWeibullData, caltime::Float64, nevents=NaN)##0.5*fpar.n.*[1-par.prev, par.prev])
    if any(isnan.(nevents))
        n = fpar.n
        nevents = Int.(floor.([n*(1-par.prev), n*par.prev]))
    else
        n = Int(floor(maximum(nevents ./ [1-par.prev, par.prev])/0.75))
    end

    ##n = fpar.n

    tmax = fpar.tmax
    max_entry = n / 15 ##fpar.max_entry

    beta = par.beta
    n0 = length(data0.time)

    z = rand(Binomial(1, 0.5), n) .- 0.5
    ##b = 1 .+ rand(Binomial(1, par.prev), n)

    b = ones(Int, n)
    b[sample(1:n, Int(floor(n*par.prev)); replace=false)] .= 2

    ##@info mean(b .== 1), mean(b .== 2), par.prev, nevents, sum(b .== 1), sum(b .== 2)

    if n < n0
        @error "requested sample size must be at least as large as sample size of old data"
    end

    entry = Vector{Float64}(undef,n)
    chaz = zeros(Float64, n)

    if caltime >= max_entry
        if n > n0
            @error "max_entry has passed, cannot recruit more patients"
        end
    else
        copyto!(entry, rand(Uniform(caltime, max_entry), n))
    end

    left = caltime .- entry

    ## keep old data
    if n0 > 0
        z[1:n0] = data0.z
        b[1:n0] = data0.b

        for i in 1:n0
            chaz[i] = par.scale[b[i]]*exp(data0.z[i]*beta[b[i]])*data0.time[i]^par.shape[b[i]]
        end

        entry[1:n0] = data0.entry
        left[1:n0] = caltime .- data0.entry
        keep = data0.time .< left[1:n0]
        chaz[1:n0][keep] = zeros(Float64, sum(keep))
    end

    U = .-log.(1 .- rand(Uniform(0, 1), n)) .+ chaz

    T = Vector{Float64}(undef, n)

    for i in 1:n
        T[i] = (U[i] / (exp(z[i]*beta[b[i]])*par.scale[b[i]]))^(1/par.shape[b[i]])
    end

    time = min.(T, tmax)
    status = T .< tmax

    max_id = 0
    id = zeros(Int, n)

    ## keep old data
    if n0 > 0
        keep = data0.time .< left[1:n0]

        if sum(keep) > 0
            keep2 = zeros(Bool, n)
            keep2[1:n0] = keep

            time[keep2] = data0.time[keep]
            status[keep2] = data0.status[keep]
        end

        id[1:n0] = data0.id
        max_id = maximum(data0.id)
    end

    if n > n0
        id[(n0+1):end] = (max_id+1):(max_id+n-n0)
    end

    BiPHWeibullData(id, entry, time, status, b, z)
end

function select_data(t::Float64, data::BiPHWeibullData; entry_stop=t)
    sel = data.entry .<= entry_stop
    entry2 = data.entry[sel]
    time2 = data.time[sel]
    status2 = data.status[sel]

    cutoff = max.(t .- entry2, 0)

    status2 = status2 .& (time2 .<= cutoff)
    time2 = min.(time2, cutoff)

    BiPHWeibullData(data.id[sel], entry2, time2, status2, data.b[sel], data.z[sel])
end

loglik(data::BiPHWeibullData, param::BiPHWeibullParam) = log_likelihood(param.shape, param.scale, param.beta, data)

function log_likelihood_b(b, shape, scale, beta, data)
    selb = data.b .== b
    time = data.time[selb]
    status = data.status[selb]
    zb = data.z[selb] * beta

    if (shape .<= 0.0) | (scale .<= 0.0)
        -Inf
    else
        sum(status.*(log(shape) + log(scale) .+ zb .+ (shape-1).*log.(time)) .- (scale .* exp.(zb) .* time.^shape))
    end
end

function log_likelihood(shape, scale, beta, prev, data)
    ll0 = log_likelihood_b(1, shape[1], scale[1], beta[1], data)
    ll1 = log_likelihood_b(2, shape[2], scale[2], beta[2], data)

    ##ll0 + ll1 + logpdf(Binomial(length(data.b), prev), sum(data.b .== 2))
    ll0 + ll1 + sum(data.b .== 1)*log(1 - prev) + sum(data.b .== 2)*log(prev)
end

logit(p) = log(p) - log(1-p)
expit(x) = exp(x) / (1+exp(x))

function laplace_approx(data::BiPHWeibullData, param::BiPHWeibullParam, prior::BiPHWeibullPrior)
    sel1 = data.b .== 1
    sel2 = data.b .== 2

    J = length(param.beta)
    beta0 = param.beta

    n1 = sum(data.b .== 1)
    n2 = sum(data.b .== 2)

    function f(x)
        ## prev, shape1, scale1, beta1, shape2, scale2, beta2
        prev = expit(x[1])
        shape1 = exp(x[2])
        shape2 = exp(x[3])
        scale1 = exp(x[4])
        scale2 = exp(x[5])
        beta1 = x[6]
        beta2 = x[7]

        -(log_likelihood((shape1, shape2), (scale1, scale2), (beta1, beta2), prev, data) + logpdf(prior.shape[1], shape1) + logpdf(prior.shape[2], shape2) + logpdf(prior.scale[1], scale1) + logpdf(prior.scale[2], scale2) + logpdf(prior.prev, prev) + log(prev) + log(1-prev) + x[1] + x[2] + x[3] + x[4] + logpdf(prior.beta, [beta1, beta2]))
    end

    res = optimize(f, [logit(0.5), log.(param.shape)..., log.(param.scale)..., param.beta...])
    est = Optim.minimizer(res)

    ## hessian of -loglik
    H = ForwardDiff.hessian(f, est)

    beta1 = est[6]
    beta2 = est[7]

    H_chol = cholesky!(H; check=false)

    if issuccess(H_chol)
        L = inv(H_chol.L)
        Sigma = (L'*L)[[6,7],[6,7]]
        [Normal(beta1, sqrt(Sigma[1,1])), Normal(beta2, sqrt(Sigma[2,2]))]
        ##MvNormal([beta1, beta2], (L'*L)[[4,7],[4,7]])
    else
        @debug "singular hessian"
        [Dirac.DiracPM(beta1), Dirac.DiracPM(beta2)]
    end
end

function get_approx(param::BiPHWeibullParam, new_data::BiPHWeibullData, prior::BiPHWeibullPrior)
    laplace_approx(new_data, param, prior)
end

function sample_param1(shape, scale, beta, prev, data, prior, V, t)
    function log_target(x)
        p = expit(x[end])
        log_likelihood(exp.(x[1:2]), exp.(x[3:4]), beta, p, data) + logpdf(prior.shape[1], exp(x[1])) + logpdf(prior.shape[2], exp(x[2])) + logpdf(prior.scale[1], exp(x[3])) + logpdf(prior.scale[2], exp(x[4])) + x[1] + x[2] + x[3] + x[4] + logpdf(prior.prev, p) + log(p) + log(1-p)
    end
    AM_RR_single(log_target, vcat(log.(shape), log.(scale), logit(prev)), V, t)
end

function sample_beta(shape, scale, beta, prev, data, prior)
    function log_target(x)
        log_likelihood(shape, scale, x, prev, data) + logpdf(prior.beta, x)
    end
    MH_single(log_target, beta, rand(MvNormal(beta, [0.01 0.0; 0.0 0.01])))
end

function sample_param2(shape, scale, beta, prev, data, prior, V, t)
    function log_target(x)
        p = expit(x[end])
        log_likelihood(exp.(x[1:2]), exp.(x[3:4]), x[5:6], p, data) + logpdf(prior.shape[1], exp(x[1])) +
            logpdf(prior.shape[2], exp(x[2])) + logpdf(prior.scale[1], exp(x[3])) + logpdf(prior.scale[2], exp(x[4])) +
            x[1] + x[2] + x[3] + x[4] + logpdf(prior.beta, x[5:6]) + logpdf(prior.prev, p) + log(p) + log(1-p)
    end

    AM_RR_single(log_target, vcat(log.(shape), log.(scale), beta, logit(prev)), V, t)
end

function fit(M::Int, data::BiPHWeibullData; prior::BiPHWeibullPrior=BiPHWeibullPrior(), warmup::Int=Int(floor(M/2)))
    n = length(data.time)

    if M == 0 | n == 0
        return BiPHWeibullFit(Matrix{Float64}(undef,0,0), Matrix{Float64}(undef,0,0), Matrix{Float64}(undef,0,0), Vector{Float64}(undef,0), 0.0, prior)
    end

    beta = Matrix{Float64}(undef,M+1,2)
    beta[1,:] = zeros(Float64, 2)
    shape = ones(Float64, M+1, 2)
    scale = ones(Float64, M+1, 2)
    prev = Vector{Float64}(undef, M+1)
    prev[1] = 0.5

    if prior.beta isa MultivariateDiracPM
        V = diagm(0 => ones(Float64, 5))
        m = [log.(shape[1,:])..., log.(scale[1,:])..., logit(prev[1])]
    else
        V = diagm(0 => ones(Float64, 5)) ##7))
        m = [log.(shape[1,:])..., log.(scale[1,:])..., logit(prev[1])]
    end

    eps = 0.001*V

    attempt = 0
    accept = 0
    ac = 0

    for t in 1:M
        attempt += 1

        if prior.beta isa MultivariateDiracPM
            par, ac = sample_param1(shape[t,:], scale[t,:], beta[t,:], prev[t], data, prior, V, t)
            beta[t+1,:] = prior.beta.x
        else
            par, ac = sample_param1(shape[t,:], scale[t,:], beta[t,:], prev[t], data, prior, V, t)
            beta[t+1,:], ac = sample_beta(exp.(par[1:2]), exp.(par[3:4]), beta[t,:], expit(par[end]), data, prior)
        end

        shape[t+1,:] = exp.(par[1:2])
        scale[t+1,:] = exp.(par[3:4])
        prev[t+1] = expit(par[end])

        new_m = (par + m*t) / (t+1)
        V = updateVar(t+1, par, new_m, m, V, eps)
        m = new_m
        accept += ac
    end

    index = (warmup+1):M
    BiPHWeibullFit(shape[index,:], scale[index,:], beta[index,:], prev[index], accept/attempt, prior)
end

function summary(x::BiPHWeibullFit)
    tmp = hcat(x.shape, x.scale, x.beta, x.prev)

    DataFrame(variable=["shape1", "shape2", "scale1", "scale2", "beta1", "beta2", "prev"],
              mean=mean(tmp,dims=1)[1,:], stderr=sqrt.(var(tmp, dims=1)[1,:]),
              neff=[effective_sample_size(tmp[:,j]) for j in 1:size(tmp,2)])
end

function fit(chains::Int, M::Int, data::BiPHWeibullData; prior::BiPHWeibullPrior=BiPHWeibullPrior(), warmup::Int=Int(floor(M/2)))
    @distributed vcat for k in 1:chains
        fit(M, data; warmup=warmup, prior=prior)
    end
end

function merge_chains(a::BiPHWeibullFit, b::BiPHWeibullFit)
    Fit(vcat(a.shape, b.shape), vcat(a.scale, b.scale), vcat(a.beta, b.beta), vcat(a.prev, b.prev), 0.5*(a.accept + b.accept), a.prior)
end

merge_chains(a) = a
merge_chains(a...) = foldr(merge_chains, a)

function summary(x::Vector{BiPHWeibullFit})
    chains = length(x)

    z = merge_chains(x...)
    df = summary(z)

    if chains > 1
        tmp = hcat(z.shape, z.scale, z.beta, z.prev)
        len = Int(floor(size(x[1].beta, 1)/2))
        df.Rhat = [potential_scale_reduction([tmp[(1+(k-1)*len):(k*len),j] for k in 1:(2*chains)]...) for j in 1:size(tmp,2)]
    end

    df
end

function sim(B::Int, M::Int; warmup=Int(floor(M/2)),
             m=BiPHWeibullModel(), prior=Prior())

    reset_timer!(to)

    J = 7
    z_mean = SharedArray{Float64,2}((B,J))
    z_var = SharedArray{Float64,2}((B,J))
    z_cp = SharedArray{Bool,2}((B,J))

    par_z = vcat(m.param.shape..., m.param.scale..., m.param.beta..., m.param.prev)

    @sync @distributed for b in 1:B
        data = generate_data(m.param, m.fparam)
        x = fit(M, data, warmup=warmup, prior=prior)
        z = hcat(x.shape, x.scale, x.beta, x.prev)

        z_mean[b,:] = mean(z, dims=1)
        z_var[b,:] = var(z, dims=1)
        z_cp[b,:] = [(quantile(z[:,j], 0.025) < par_z[j]) .& (quantile(z[:,j], 0.975) > par_z[j]) for j in 1:length(par_z)]
    end

    DataFrame(variable=["shape1", "shape2", "scale1", "scale2", "beta1", "beta2", "prev"],
              mean=mean(z_mean, dims=1)[1,:],
              sse=sqrt.(var(z_mean, dims=1)[1,:]),
              ese=sqrt.(mean(z_var, dims=1)[1,:]),
              cp=mean(z_cp, dims=1)[1,:])
end


function biphweibull_test_chains(M, chains)
    model = BiPHWeibullModel0()
    data = model.generate_data(model.param, model.fparam)
    x = fit(chains, M, data)
    summary(x)
end

function biphweibull_test_laplace_approx()
    m = BiPHWeibullModel0()
    data = m.generate_data(m.param, m.fparam)

    la = laplace_approx(data, m.param, BiPHWeibullPrior())
    x = fit(10000, data; warmup=5000)

    print(BiPHWeibull.summary(x))

    mean(la[1]), mean(la[2]), sqrt(var(la[1])), sqrt(var(la[2])), mean(x.beta, dims=1), sqrt.(var(x.beta, dims=2))
end
