struct BiPHSplineData <: AbstractData
    id::Vector{Int}
    entry::Vector{Float64}
    time::Vector{Float64}
    status::Vector{Bool}
    b::Vector{Int}
    z::Vector{Float64}
end

BiPHSplineData() = BiPHSplineData(Vector{Int}(undef,0), Vector{Float64}(undef,0), Vector{Float64}(undef,0),
Vector{Bool}(undef,0), Vector{Int}(undef,0), Vector{Float64}(undef,0))

mutable struct BiPHSplineParam <: AbstractParam
    spline::NTuple{2,Spline1D}
    beta::NTuple{2,Float64}
    prev::Float64 ## biomarker-positive subgroup prevalence
end

struct BiPHSplineFixedParam <: AbstractFixedParam
    n::Int
    tmax::Float64
    max_entry::Float64
end

struct BiPHSplinePrior <: AbstractPrior
    spline::NTuple{2,SplineHazard.Prior}
    beta::MultivariateDistribution
    prev::UnivariateDistribution
end

BiPHSplinePrior() = BiPHSplinePrior((SplineHazard.Prior(), SplineHazard.Prior()), MvNormal(10*diagm(0 => ones(Float64, 2))), Beta(2, 2))

BiPHSplineModel() = Model(BiPHSplineParam((SplineHazard.Spline.const_spline(1.0, (0.0, 36.0)), SplineHazard.Spline.const_spline(1.0, (0.0, 36.0))),
                                (log(0.95), log(0.83)), 0.4), FixedParam(400, 36.0, 24.0), generate_data)

BiPHSplineModel(param, fparam, gen_data=generate_data) = Model(param, fparam, gen_data)


struct BiPHSplineFit <: AbstractFit
    s::NTuple{2,SplineHazard.Sampler}
    beta::Array{Float64,2} ## hazard ratios
    prev::Vector{Float64} ## biomarker-positive subgroup prevalence
    accept::Float64 ## acceptance rates
    prior::Prior
end

nevents(data::BiPHSplineData) = sum(data.status)

function slice(i::Int, x::BiPHSplineFit)
    BiPHSplineParam((SplineHazard.extract_spline(x.s[1], i), SplineHazard.extract_spline(x.s[2], i)), (x.beta[i,1], x.beta[i,2]), x.prev[i])
end

function sample_prior(B::Int, prior::BiPHSplinePrior)
    beta = rand(prior.beta, B)
    prev = rand(prior.prev, B)

    [BiPHSplineParam((SplineHazard.sample_prior(first(prior.spline)), SplineHazard.sample_prior(last(prior.spline))),
           (beta[1,b], beta[2,b]), prev[b]) for b in 1:B]
end

## return prior distribution (before sampling) or posterior sample (after sampling)
## of treatment effect parameter
function get_outcome(x::BiPHSplineFit)
    if size(x.beta)[1] == 0
        x.prior.beta
    else
        x.beta
    end
end

function inv_cumhaz(U::Float64, beta, group, s::Spline1D, tmax::Float64)
    f(x) = SplineHazard.Spline.cumhaz(x, s)*exp(group*beta) - U

    if f(tmax) < 0
        Inf
    else
        fzero(f, 0, tmax)
    end
end

function generate_data(par::BiPHSplineParam, fpar::BiPHSplineFixedParam)
    ##predict_data(par, fpar, Data(), 0.0)

    n = fpar.n
    J = length(par.beta)

    z = rand(Binomial(1, 0.5), n) .- 0.5
    b = 1 .+ rand(Binomial(1, par.prev), n)

    ## 25 patients per month
    max_entry = fpar.tmax ##/ 25
    entry = rand(Uniform(0, fpar.max_entry), n)
    ##U = -log.(1 .-rand(Uniform(0, 1), n))
    U = 1 .- rand(Uniform(0, 1), n)

    T = exp.([quantile(Logistic(log(12), 1/6), 1 .- exp.(log(U[i]).*exp.(-par.beta[b[i]]*z[i]))) for i in 1:n])

    ##C = min.(rand(Exponential(fpar.tmax), n), fpar.tmax)
    C = min.(rand(Uniform(0, 3*36), n), fpar.tmax)

    time = min.(T, C)
    status = T .< C
    id = 1:fpar.n

    BiPHSplineData(id, entry, time, status, b, z)
end

function predict_data(par::BiPHSplineParam, fpar::BiPHSplineFixedParam, data0::BiPHSplineData, caltime::Float64, nevents=NaN)
    if any(isnan.(nevents))
        n = fpar.n
        N = n.*[1-par.prev, par.prev]
    else
        N = Int.(floor.(nevents ./ [1-par.prev, par.prev]))
        n = sum(N)
    end

    tmax = fpar.tmax
    max_entry = n / 15 ##fpar.max_entry

    beta = par.beta
    n0 = length(data0.time)

    z = rand(Binomial(1, 0.5), n) .- 0.5
    b = 1 .+ rand(Binomial(1, par.prev), n)

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

        chaz[1:n0] = [SplineHazard.Spline.cumhaz(data0.time[i], par.spline[b[i]]).*exp.(z[i] * beta[b[i]]) for i in 1:n0]
        entry[1:n0] = data0.entry
        left[1:n0] = caltime .- data0.entry
        keep = data0.time .< left[1:n0]
        chaz[1:n0][keep] = zeros(Float64, sum(keep))
    end

    U = .-log.(1 .- rand(Uniform(0, 1), n)) .+ chaz
    T = [inv_cumhaz(U[i], beta[b[i]], z[i], par.spline[b[i]], 500.0) for i in 1:n]

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

    BiPHSplineData(id, entry, time, status, b, z)
end

function select_data(t::Float64, data::BiPHSplineData; entry_stop=t)
    sel = data.entry .<= entry_stop
    entry2 = data.entry[sel]
    time2 = data.time[sel]
    status2 = data.status[sel]

    cutoff = max.(t .- entry2, 0) .+ 1e-14

    status2 = status2 .& (time2 .<= cutoff)
    time2 = min.(time2, cutoff)

    Data(data.id[sel], entry2, time2, status2, data.b[sel], data.z[sel])
end

function loglik(data::BiPHSplineData, param::BiPHSplineParam)
    log_likelihood(param.spline, param.beta, data)
end

function log_likelihood_b(b::Int, s::Spline1D, beta, theta, data::BiPHSplineData,
                          use_precalc=false, precalc_haz=0.0, precalc_cumhaz=0.0)

    selb = data.b .== b
    time = data.time[selb]
    status = data.status[selb]
    zb = data.z[selb] * beta

    if use_precalc
        h = precalc_haz * exp.(theta)
        ch = precalc_cumhaz * exp.(theta)
    else
        h = SplineHazard.Spline.hazard(time[status], s)
        ch = SplineHazard.Spline.cumhaz(time, s)
    end

    sum(log.(h) .+ zb[status]) - sum(ch.*exp.(zb))
end

function log_likelihood(s::NTuple{2,Spline1D}, beta, prev, theta1, theta2, data::BiPHSplineData, use_precalc=false,
                        precalc_haz1=0.0, precalc_cumhaz1=0.0,
                        precalc_haz2=0.0, precalc_cumhaz2=0.0)

    ll0 = log_likelihood_b(1, s[1], beta[1], theta1, data, use_precalc, precalc_haz1, precalc_cumhaz1)
    ll1 = log_likelihood_b(2, s[2], beta[2], theta2, data, use_precalc, precalc_haz2, precalc_cumhaz2)

    ll0 + sum(data.b .== 1)*log(1 - prev) + ll1 + sum(data.b .== 2)*log(prev)
end

function laplace_approx(data::BiPHSplineData, param::BiPHSplineParam, prior::BiPHSplinePrior)
    sel1 = data.b .== 1
    sel2 = data.b .== 2

    precalc_haz1, precalc_cumhaz1 = SplineHazard.Spline.eval_basis(param.spline[1], data.time[data.status .& sel1], data.time[sel1])
    precalc_haz2, precalc_cumhaz2 = SplineHazard.Spline.eval_basis(param.spline[2], data.time[data.status .& sel2], data.time[sel2])

    J, K1, K2 = length(param.beta), size(precalc_haz1)[2], size(precalc_haz2)[2]

    theta10 = log.(param.spline[1].c)
    theta20 = log.(param.spline[2].c)
    beta0 = param.beta

    function f(x)
        ## prev, v1, theta1, beta1, v2, theta2, beta2
        prev = expit(x[1])
        v1 = exp(2*x[2])
        theta1 = x[3:(K1+2)]
        beta1 = x[K1+3]

        v2 = exp(2*x[K1+4])
        theta2 = x[(K1+4+1):(K1+4+K2)]
        beta2 = x[end]

        -(log_likelihood(param.spline, [beta1, beta2], prev, theta1, theta2, data, true,
                         precalc_haz1, precalc_cumhaz1, precalc_haz2, precalc_cumhaz2)
          - dot(theta1,theta1)/(2*v1) - K1*log(v1)/2 - dot(theta2,theta2)/(2*v2) - K2*log(v2)/2) + logpdf(prior.prev, prev) + log(prev) + log(1-prev) + x[2] + x[K1+4]
    end

    res = optimize(f, [logit(0.5), 0.0, theta10..., beta0[1], 0.0, theta20..., beta0[2]])

    @show res

    est = Optim.minimizer(res)
    H = Symmetric(ForwardDiff.hessian(f, est))

    beta1 = est[K1+3]
    beta2 = est[end]

    H_chol = cholesky!(H; check=false)

    if issuccess(H_chol)
        L = inv(H_chol.U')
        Sigma = (L'*L)[[K1+3,end],[K1+3,end]]
        [Normal(beta1, sqrt(Sigma[1,1])), Normal(beta2, sqrt(Sigma[2,2]))]
        ##MvNormal([beta1, beta2], (L'*L)[[4,7],[4,7]])
    else
        [Dirac.DiracPM(beta1), Dirac.DiracPM(beta2)]
    end
end

function sample_param1(s::NTuple{2,Spline1D}, beta, prev, data, prior, V, t)
    function log_target(x)
        p = expit(x[1])
        log_likelihood(s, beta, p, [0.0], [0.0], data) + logpdf(prior.prev, p) + log(p) + log(1-p)
    end

    AM_RR_single(log_target, [logit(prev)], V, t)
end

function sample_param2(s::NTuple{2,Spline1D}, beta, prev, data, prior, V, t)
    function log_target(x)
        p = expit(x[3])
        log_likelihood(s, x[1:2], p, [0.0], [0.0], data) + logpdf(prior.beta, x[1:2]) + logpdf(prior.prev, p) + log(p) + log(1-p)
    end

    AM_RR_single(log_target, vcat(beta, logit(prev)), V, t)
end

function get_approx(param::BiPHSplineParam, new_data::BiPHSplineData, prior::BiPHSplinePrior)
    ##laplace_approx(new_data, param, prior)

    x = fit(5000, new_data; warmup=2000, prior=prior)
    [Normal(mean(x.beta[:,1]), sqrt(var(x.beta[:,1]))), Normal(mean(x.beta[:,2]), sqrt(var(x.beta[:,2])))]

    # S = cov(x.beta)
    # S_chol = cholesky!(S; check=false)

    # if !issuccess(S_chol)
    #     Dirac.MultivariateDiracPM(mean(x.beta, dims=1)[1,:])
    # else
    #     MvNormal(mean(x.beta, dims=1)[1,:], Distributions.PDMats.PDMat(S_chol))
    # end
end

logit(p) = log(p) - log(1-p)
expit(x) = exp(x) / (1+exp(x))

function fit(M::Int, data::BiPHSplineData; prior::BiPHSplinePrior=BiPHSplinePrior(), warmup::Int=Int(floor(M/2)))
    n = length(data.time)

    if M == 0 || n == 0
        s1 = SplineHazard.create_sampler((sp, x) -> log_likelihood_b(1, sp, x, [0.0], data))
        s2 = SplineHazard.create_sampler((sp, x) -> log_likelihood_b(2, sp, x, [0.0], data))
        return Fit((s1, s2), Matrix{Float64}(undef,0,0), Vector{Float64}(undef,0), 0.0, prior)
    end

    s1 = SplineHazard.setup_sampler(M, data.time[(data.b .== 1) .& data.status], (sp, x) -> log_likelihood_b(1, sp, x, [0.0], data))
    s2 = SplineHazard.setup_sampler(M, data.time[(data.b .== 2) .& data.status], (sp, x) -> log_likelihood_b(2, sp, x, [0.0], data))

    sp1 = SplineHazard.extract_spline(s1, 1)
    sp2 = SplineHazard.extract_spline(s2, 1)

    J = length(prior.beta)
    beta = Array{Float64,2}(undef,M+1,J)
    beta[1,:] = zeros(Float64, J)

    prev = Vector{Float64}(undef, M+1)
    prev[1] = 0.5

    if prior.beta isa MultivariateDiracPM
        V = diagm(0 => ones(Float64, 1))
        m = [logit(prev[1])]
    else
        V = diagm(0 => ones(Float64, J+1))
        m = vcat(beta[1,:], logit(prev[1]))
    end

    eps = 0.001*V

    attempt = 0
    accept = 0
    ac = 0

    for t in 1:M
        attempt += 1

        if prior.beta isa MultivariateDiracPM
            par, ac = sample_param1((sp1, sp2), beta[t,:], prev[t], data, prior, V, t)
            beta[t+1,:] = prior.beta.x
        else
            par, ac = sample_param2((sp1, sp2), beta[t,:], prev[t], data, prior, V, t)
            beta[t+1,:] = par[1:2]
        end

        prev[t+1] = expit(par[end])
        new_m = (par + m*t) / (t+1)
        V = updateVar(t+1, par, new_m, m, V, eps)
        m = new_m
        accept += ac

        SplineHazard.update!(s1, beta[t+1,1])
        sp1 = SplineHazard.extract_spline(s1, t)

        SplineHazard.update!(s2, beta[t+1,2])
        sp2 = SplineHazard.extract_spline(s2, t)
    end

    index = (warmup+1):M
    BiPHSplineFit((s1, s2), beta[index,:], prev[index], accept/attempt, prior)
end

function summary(x::BiPHSplineFit)
    B, J = size(x.beta)

    ##SplineHazard.summary(x, time)

    DataFrame(variable=[repeat(["beta"], inner=J)..., "prev"],
              mean=[mean(x.beta,dims=1)[1,:]..., mean(x.prev)],
              stderr=sqrt.([var(x.beta,dims=1)[1,:]..., var(x.prev)]),
              neff=[[effective_sample_size(x.beta[:,j]) for j in 1:J]..., effective_sample_size(x.prev)])
end

function fit(chains::Int, M::Int, data::BiPHSplineData; prior::BiPHSplinePrior=BiPHSplinePrior(), warmup::Int=Int(floor(M/2)))
    @distributed vcat for k in 1:chains
        fit(M, data; warmup=warmup, prior=prior)
    end
end

function merge_chains(a::BiPHSplineFit, b::BiPHSplineFit)
    Fit(a.s, vcat(a.beta, b.beta), vcat(a.prev, b.prev), 0.5*(a.accept + b.accept), a.prior)
end

merge_chains(a) = a
merge_chains(a...) = foldr(merge_chains, a)

function summary(x::Vector{BiPHSplineFit})
    chains = length(x)

    z = merge_chains(x...)
    df = summary(z)

    if chains > 1
        tmp = hcat(z.beta, z.prev)
        len = Int(floor(size(x[1].beta, 1)/2))
        df.Rhat = [potential_scale_reduction([tmp[(1+(k-1)*len):(k*len),j] for k in 1:(2*chains)]...) for j in 1:size(tmp,2)]
    end

    df
end

function sim(B::Int, M::Int; warmup=Int(floor(M/2)),
             m=BiPHSplineModel(), prior=BiPHSplinePrior())
    res = Array{Float64, 2}(undef,B, 4)

    J = length(m.param.beta)
    beta = SharedArray{Float64,2}((B,J))
    var_beta = SharedArray{Float64,2}((B,J))
    cpb = SharedArray{Bool,2}((B,J))

    prev = SharedArray{Float64,1}(B)
    var_prev = SharedArray{Float64,1}(B)
    cpp = SharedArray{Bool,1}(B)

    @sync @distributed for b in 1:B
        data = generate_data(m.param, m.fparam)
        data = select_data(m.fparam.tmax, data)
        x = fit(M, data, warmup=warmup, prior=prior)

        beta[b,:] = mean(x.beta, dims=1)
        var_beta[b,:] = var(x.beta, dims=1)
        cpb[b,:] = [(quantile(x.beta[:,j], 0.025) < m.param.beta[j]) .& (quantile(x.beta[:,j], 0.975) > m.param.beta[j]) for j in 1:length(m.param.beta)]

        prev[b,:] = mean(x.prev, dims=1)
        var_prev[b,:] = var(x.prev, dims=1)
        cpp[b,:] = [(quantile(x.prev, 0.025) < m.param.prev) .& (quantile(x.prev, 0.975) > m.param.prev) for j in 1:length(m.param.prev)]
    end

    DataFrame(variable=[repeat(["beta"], inner=J)..., "prev"],
              mean=[mean(beta,dims=1)[1,:]..., mean(prev)],
              sse=sqrt.([var(beta,dims=1)[1,:]..., var(prev)]),
              ese=sqrt.([mean(var_beta,dims=1)[1,:]..., mean(var_prev)]),
              cp=[mean(cpb,dims=1)[1,:]..., mean(cpp)])
end

function biphspline_test_chains(M, chains)
    model = BiPHSplineModel()
    data = model.generate_data(model.param, model.fparam)
    x = fit(chains, M, data)
    summary(x)
end

function biphspline_test_laplace_approx()
    m = BiPHSplineModel()
    data = m.generate_data(m.param, m.fparam)

    la = laplace_approx(data, m.param, BiPHSplinePrior())
    x = fit(5000, data; warmup=2000)

    df = summary(x)
    la, df
end
