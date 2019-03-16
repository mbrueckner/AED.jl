## Simple PH model with Weibull baseline hazard

struct WeibullPHData <: AbstractData
    id::Vector{Int}
    entry::Vector{Float64}
    time::Vector{Float64}
    status::Vector{Bool}
    z::Matrix{Float64}
end

WeibullPHData() = WeibullPHData(Vector{Int}(undef, 0), Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
              Vector{Bool}(undef, 0), Matrix{Float64}(undef, 0, 0))

mutable struct WeibullPHParam{T <: Real} <: AbstractParam
    shape::T
    scale::T
    beta::Vector{T}
end

struct WeibullPHFixedParam <: AbstractFixedParam
    n::Int
    tmax::Float64
    max_entry::Float64
end

struct WeibullPHPrior <: AbstractPrior
    shape::ContinuousUnivariateDistribution
    scale::ContinuousUnivariateDistribution
    beta::MultivariateDistribution
end

WeibullPHPrior(k=1) = WeibullPHPrior(Gamma(1, 1), Gamma(1, 1), MvNormal(diagm(0 => 10*ones(k))))

## exponential model
WeibullPHModel() = Model(WeibullPHParam(1.0, 1.0, [0.0]), WeibullPHFixedParam(100, 55.0, 48.0), ph_exp_1)

WeibullPHModel(param, fparam, gen_data=ph_exp_1) = Model(param, fparam, gen_data)

loglik(data::WeibullPHData, param::WeibullPHParam) = log_likelihood(param.shape, param.scale, param.beta, data)

struct WeibullPHFit <: AbstractFit
    shape::Vector{Float64} ## Weibull shape
    scale::Vector{Float64} ## Weibull scale
    beta::Matrix{Float64} ## hazard ratio
    accept::Float64 ## acceptance rates
    prior::Prior
end

WeibullPHFit() = WeibullPHFit(Vector{Float64}(undef,0), Vector{Float64}(undef,0), Matrix{Float64}(undef,0), Vector{Float64}(undef,0), Prior())

function slice(i::Int, x::WeibullPHFit)
    if length(x.shape) == 0
        sample_prior(1, x.prior)[1]
    else
        WeibullPHParam(x.shape[i], x.scale[i], x.beta[i,:])
    end
end

function sample_prior(B::Int, prior::WeibullPHPrior)
    shape = rand(prior.shape, B)
    scale = rand(prior.scale, B)
    beta = rand(prior.beta, B)
    [WeibullPHParam(shape[b], scale[b], beta[:,b]) for b in 1:B]
end

## return prior distribution (before sampling) or posterior sample (after sampling)
## of treatment effect parameter
function get_outcome(x::WeibullPHFit)
    if size(x.beta, 1) == 0
        x.prior.beta
    else
        x.beta
    end
end

function predict_data(par::WeibullPHParam, fpar::WeibullPHFixedParam, data0::WeibullPHData, caltime::Float64, nevents=NaN)
    n = fpar.n
    tmax = fpar.tmax
    max_entry = n / 15 ##fpar.max_entry

    beta = par.beta
    shape = par.shape
    scale = par.scale
    n0 = length(data0.time)

    J = length(par.beta)
    group = Matrix{Float64}(undef,n,J)
    z = sample(0:J, n)

    for j in 1:J
        group[:,j] = (z .== j) .- 0.5
    end

    if n < n0
        error("requested sample size must be at least as large as sample size of old data")
    end

    entry = Vector{Float64}(undef, n)
    chaz = zeros(Float64, n)

    if caltime >= max_entry
        if n > n0
            error("max_entry has passed, cannot recruit more patients")
        end
    else
        copyto!(entry, rand(Uniform(caltime, max_entry), n))
    end

    left = caltime .- entry

    ## keep old data
    if n0 > 0
        for j in 1:J
            group[1:n0,j] = data0.z[:,j]
        end
        entry[1:n0] = data0.entry
        chaz[1:n0] = scale*exp.(data0.z*beta).*data0.time.^shape
        left[1:n0] = caltime .- data0.entry
        keep = data0.time .< left[1:n0]
        chaz[1:n0][keep] .= 0.0
    end

    U = .-log.(1 .-rand(Uniform(0, 1), n)) .+ chaz
    T = (U ./ (exp.(group*beta).*scale)).^(1/shape)

    time = min.(T, tmax)
    status = T .< tmax

    max_id = 0
    id = zeros(Int, n)

    ## keep old data
    if n0 > 0
        keep = data0.time .< left[1:n0]

        if sum(keep) > 0
            keep2 = Vector{Bool}(undef, n)
            fill!(keep2, false)
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

    WeibullPHData(id, entry, time, status, group)
end

function ph_exp_1(par::WeibullPHParam, fpar::WeibullPHFixedParam)
    n = fpar.n

    z = zeros(Float64, n, 1)
    z[:,1] = rand(Binomial(1, 0.5), n) .- 0.5

    entry = rand(Uniform(0, fpar.max_entry), n)
    U = .-log.(1 .-rand(Uniform(0, 1), n))
    T = (U./(par.scale .* exp.(z*par.beta))).^(1/par.shape)
    ##T = U.^(1/par.shape).*exp.(-par.beta*z).*par.scale
    ##C = min.(rand(Exponential(fpar.tmax), n), fpar.tmax)
    C = min.(rand(Exponential(fpar.tmax), n), fpar.tmax)

    time = min.(T, C)
    status = T .< C
    id = 1:fpar.n

    return Data(id, entry, time, status, z)
end

function ph_weibull_1(par, fpar)
    n = fpar.n

    z = rand(Binomial(1, 0.5), n) .- 0.5

    ## 25 patients per month
    ##max_entry = n / 25
    entry = rand(Uniform(0, fpar.max_entry), n)
    ##U = -log.(1 .-rand(Uniform(0, 1), n))
    U = 1 .- rand(Uniform(0, 1), n)

    shape = 1.25
    scale = log(2)/22.5

    T = [quantile(Gamma(6, 2), 1 .- exp.(log(U[i]).*exp.(.-par.beta .* z[i]))) for i in 1:n]

    ##C = min.(rand(Exponential(fpar.tmax), n), fpar.tmax)
    C = min.(rand(Uniform(0, 3*36), n), fpar.tmax)

    time = min.(T, C)
    status = T .< C
    id = 1:fpar.n

    Data(id, entry, time, status, z)
end

function select_data(t::Float64, data::WeibullPHData; entry_max=t)
    sel = data.entry .<= entry_max
    entry2 = data.entry[sel]
    time2 = data.time[sel]
    status2 = data.status[sel]

    cutoff = max.(t .- entry2, 0)

    status2 = status2 .& (time2 .<= cutoff)
    time2 = min.(time2, cutoff)

    Data(data.id[sel], entry2, time2, status2, data.z[sel,:])
end

S(time, z, lambda, beta) = exp.(-(time ./ (scale .* exp.(-z*beta)))^shape)

function log_likelihood(shape, scale, beta, data)
    time = data.time
    status = data.status
    psi = exp.(data.z*beta)

    if (shape .<= 0.0) | (scale .<= 0.0)
        -Inf
    else
        sum(status.*(log(shape) + log(scale) .+ log.(psi) .+ (shape-1).*log.(time)) .- (scale .* psi .* time.^shape))
    end
end

function laplace_approx(data::WeibullPHData, param::WeibullPHParam, prior::WeibullPHPrior)
    f(x) = -log_likelihood(exp(x[1]), exp(x[2]), x[3:end], data) - logpdf(prior.shape, exp(x[1])) - x[1] - logpdf(prior.scale, exp(x[2])) - x[2] - logpdf(prior.beta, x[3:end])

    res = optimize(f, [log(param.shape), log(param.scale), param.beta...])
    x = Optim.minimizer(res)
    est_log_shape = x[1]
    est_log_scale = x[2]
    est_beta = x[3:end]

    ## hessian of -loglik
    H = ForwardDiff.hessian(f, x)

    H_chol = cholesky!(H; check=false)
    if !issuccess(H_chol)
        [DiracPM(est_beta[1])]
    else
        L = inv(H_chol.U')
        ##MvNormal(est_beta, (L'*L)[3:end,3:end])
        [Normal(est_beta[1], sqrt((L'*L)[end,end]))]
    end
end

function sample_param(shape, scale, beta, data, V, t, prior)
    target(x) = log_likelihood(exp(x[1]), exp(x[2]), x[3:end], data) + logpdf(prior.shape, exp(x[1])) + x[1] + logpdf(prior.scale, exp(x[2])) + x[2] + logpdf(prior.beta, x[3:end])
    AM_RR_single(target, [shape, scale, beta...], V, t)
end

function sample_param(shape, scale, data, V, t, prior)
    target(x) = log_likelihood(exp(x[1]), exp(x[2]), prior.beta.x, data) + logpdf(prior.shape, exp(x[1])) + x[1] + logpdf(prior.scale, exp(x[2])) + x[2]
    AM_RR_single(target, [shape, scale], V, t)
end

function get_approx(param::WeibullPHParam, new_data::WeibullPHData, prior::Prior)
    laplace_approx(new_data, param, prior)
    ##x = fit(10000, new_data; prior=prior)
    ##x.beta
    ##Normal(mean(x.beta), sqrt(var(x.beta)))
end

function fit(M::Int, data::WeibullPHData; prior::WeibullPHPrior=WeibullPHPrior(), warmup::Int=Int(floor(M/2)), use_NUTS=false)
    if M == 0 | length(data.time) == 0
        return WeibullPHFit(Vector{Float64}(undef, 0), Vector{Float64}(undef, 0), Matrix{Float64}(undef, 0, 0), 0.0, prior)
    end

    J = length(prior.beta)
    beta = Matrix{Float64}(undef,M+1,J)
    beta[1,:] = zeros(Float64, J)
    shape = ones(Float64, M+1)
    scale = ones(Float64, M+1)

    if prior.beta isa MultivariateDiracPM
        V = diagm(0 => ones(Float64, 2))
        mb = [log(shape[1]), log(scale[1])]
    else
        V = diagm(0 => ones(Float64, 2+J))
        mb = [log(shape[1]), log(scale[1]), beta[1,:]...]
    end

    eps = 0.001*V
    attempt = 0
    accept = 0
    index = (warmup+1):M

    for t in 1:M
        attempt += 1
        if prior.beta isa MultivariateDiracPM
            x, ac = sample_param(log(shape[t]), log(scale[t]), data, V, t, prior)
            beta[t+1,:] = prior.beta.x
        else
            x, ac = sample_param(log(shape[t]), log(scale[t]), beta[t,:], data, V, t, prior)
            beta[t+1,:] = x[3:end]
        end

        shape[t+1] = exp(x[1])
        scale[t+1] = exp(x[2])

        new_mb = (x + mb*t) / (t+1)
        V = updateVar(t+1, x, new_mb, mb, V, eps)
        mb = new_mb
        accept += ac
    end

    WeibullPHFit(shape[index], scale[index], beta[index,:], accept/attempt, prior)
end

function fit(chains::Int, M::Int, data::WeibullPHData; prior::WeibullPHPrior=WeibullPHPrior(), warmup::Int=Int(floor(M/2)), use_NUTS=false)
    @distributed vcat for k in 1:chains
        fit(M, data; warmup=warmup, prior=prior, use_NUTS=use_NUTS)
    end
end

function merge_chains(a::WeibullPHFit, b::WeibullPHFit)
    WeibullPHFit(vcat(a.shape, b.shape), vcat(a.scale, b.scale), vcat(a.beta, b.beta), 0.5*(a.accept .+ b.accept), a.prior)
end

merge_chains(a) = a
merge_chains(a...) = foldr(merge_chains, a)

function summary(x::Vector{WeibullPHFit})
    chains = length(x)

    z = merge_chains(x...)
    df = summary(z)

    if chains > 1
        len = Int(floor(length(x[1].shape)/2))
        df.Rhat = [potential_scale_reduction([z.beta[(1+(k-1)*len):(k*len), j] for k in 1:(2*chains) for j in 1:size(z.beta,2)]...),
                   potential_scale_reduction([z.shape[(1+(k-1)*len):(k*len)] for k in 1:(2*chains)]...),
                   potential_scale_reduction([z.scale[(1+(k-1)*len):(k*len)] for k in 1:(2*chains)]...)]
    end

    df
end

function summary(x::WeibullPHFit)
    J = size(x.beta,2)

    DataFrame(variable=[repeat(["beta"], inner=J)..., "shape", "scale"],
              mean=[mean(x.beta,dims=1)[1,:]..., mean(x.shape), mean(x.scale)],
              variance=[var(x.beta,dims=1)[1,:]..., var(x.shape), var(x.scale)],
              neff=[[effective_sample_size(x.beta[:,j]) for j in 1:J]..., effective_sample_size(x.shape), effective_sample_size(x.scale)])
end

function sim(B::Int, M::Int; warmup=Int(floor(M/2)), use_NUTS=false,
             par::WeibullPHParam=WeibullPHParam(1.2, 1.5, [log(0.77)]),
             fpar::WeibullPHFixedParam=WeibullPHFixedParam(100, 55, 48))
    res = Array{Float64, 2}(undef, B, 4)

    J = length(par.beta)
    beta = Matrix{Float64}(undef, B,J)
    var_beta = Matrix{Float64}(undef,B,J)
    cpb = Matrix{Bool}(undef, B,J)

    shape = Vector{Float64}(undef, B)
    scale = Vector{Float64}(undef, B)
    var_shape = Vector{Float64}(undef, B)
    var_scale = Vector{Float64}(undef, B)
    cpsh = Vector{Bool}(undef, B)
    cpsc = Vector{Bool}(undef, B)

    covered(x, v, b) = ((x - 1.96*sqrt(v)) < b) & (b < (x + 1.96*sqrt(v)))

    prior = Prior(Gamma(1, 1), Gamma(1, 1), MultivariateDiracPM([0.0]))

    for b in 1:B
        data = ph_exp_1(par, fpar)
        data = select_data(fpar.tmax, data)
        x = fit(M, data, prior=prior, warmup=warmup, use_NUTS=use_NUTS)

        ##@show x.accept
        ##@info effective_sample_size(x.beta), effective_sample_size(x.shape), effective_sample_size(x.scale)

        beta[b,:] = mean(x.beta, dims=1)
        var_beta[b,:] = var(x.beta, dims=1)
        cpb[b,:] = [(quantile(x.beta[:,j], 0.025) < par.beta[j]) .& (quantile(x.beta[:,j], 0.975) > par.beta[j]) for j in 1:length(par.beta)]

        shape[b] = mean(x.shape)
        var_shape[b] = var(x.shape)
        cpsh[b] = covered(shape[b], var_shape[b], par.shape)

        scale[b] = mean(x.scale)
        var_scale[b] = var(x.scale)
        cpsc[b] = covered(scale[b], var_scale[b], par.scale)
    end

    DataFrame(variable=["shape", "scale", repeat(["beta"], inner=J)...],
              mean=[mean(shape), mean(scale), mean(beta,dims=1)[1,:]...],
              sse=sqrt.([var(shape), var(scale), var(beta,dims=1)[1,:]...]),
              ese=sqrt.([mean(var_shape), mean(var_scale), mean(var_beta,dims=1)[1,:]...]),
              cp=[mean(cpsh), mean(cpsc), mean(cpb,dims=1)[1,:]...])
end

function weibull_test_laplace_approx()
    par = WeibullPH.Param(1.5, 1.0, [log(0.78)])
    fpar = WeibullPH.FixedParam(400, 55.0, 48.0)
    data = WeibullPH.ph_exp_1(par, fpar)

    x = WeibullPH.fit(10000, data)
    la = WeibullPH.laplace_approx(data, par, WeibullPH.Prior())

    @show WeibullPH.summary(x)
    @show (mean(la), var(la))
end

function weibull_test_chains(M1, M2, chains)
    model = WeibullPH.WeibullPHModel()

    data = model.generate_data(model.param, model.fparam)
    x = WeibullPH.fit(chains, M1, data, use_NUTS=false)
    summary(x)
end
