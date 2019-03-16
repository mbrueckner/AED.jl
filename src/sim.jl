## generate data from a PH model with constant baseline hazard log(2)/12
## return object of type Ph.Data, PhSpline.Data or WeibullPh.Data
function ph_exp_1(par, fpar, data_type=WeibullPHData)
    n = fpar.n

    z = zeros(Float64, n, 1)
    z[:,1] = rand(Binomial(1, 0.5), n) .- 0.5

    ## 10 patients per time unit
    max_entry = n / 15 ##fpar.max_entry ##/ 10

    entry = rand(Uniform(0, max_entry), n)
    U = -log.(1 .-rand(Uniform(0, 1), n))

    ## 12 time units median survival time
    T = (U./(par.scale .* exp.(z*par.beta))).^(1/par.shape)
    ##T = U.*exp.(-par.beta*z)./(log(2)/12)

    ## approx: 10% admin cens, 10% drop-out
    C = rand(Exponential(45/log(2)), n)
    ##C = min.(rand(Uniform(0, 3*36), n), fpar.tmax)

    time = min.(T, C)
    status = T .< C
    id = 1:fpar.n

    data_type(id, entry, time, status, z)
end

function ph_biweibull_1(par, fpar, data_type=BiPHWeibullData)
    n = fpar.n

    z = rand(Binomial(1, 0.5), n) .- 0.5
    b = 1 .+ rand(Binomial(1, par.prev), n)

    ## 25 patients per month
    max_entry = n / 15 ## fpar.max_entry
    entry = rand(Uniform(0, max_entry), n)

    U = -log.(1 .-rand(Uniform(0, 1), n))
    T = zeros(Float64, n)

    for i in 1:n
        T[i] = (U[i] / (exp(z[i]*par.beta[b[i]])*par.scale[b[i]]))^(1/par.shape[b[i]])
    end

    C = rand(Exponential(45/log(2)), n)
    ##C = min.(rand(Uniform(0, 3*36), n), fpar.tmax)

    time = min.(T, C)
    status = T .< C
    id = 1:fpar.n

    data_type(id, entry, time, status, b, z)
end

function ph_billog_1(par, fpar, data_type=BiPHSplineData)
    n = fpar.n
    z = rand(Binomial(1, 0.5), n) .- 0.5
    b = 1 .+ rand(Binomial(1, par.prev), n)

    ## 25 patients per month
    max_entry = n / 15 ## fpar.max_entry
    entry = rand(Uniform(0, max_entry), n)

    T = zeros(Float64, n)
    U = 1 .- rand(Uniform(0, 1), n)

    for i in 1:n
        T[i] = exp(quantile(Logistic(log(12), 1/6), 1 - exp(log(U[i])*exp(-par.beta[b[i]]*z[i]))))
    end

    ##C = min.(rand(Exponential(fpar.tmax), n), fpar.tmax)
    C = min.(rand(Uniform(0, 3*36), n), fpar.tmax)

    time = min.(T, C)
    status = T .< C
    id = 1:fpar.n

    data_type(id, entry, time, status, b, z)
end

function getWeibullPHModel(max_n, delta0=[log(1.0), log(0.6)], delta=log(0.6); gen_data=ph_exp_1, tmax=Inf, emax=Inf)
    dshape = Uniform(0.25, 10)
    dscale = Gamma(1, 1)

    p = WeibullPHPrior(dshape, dscale, MvNormal(diagm(0 => 100*ones(eltype(delta), 1))))
    p0 = WeibullPHPrior(dshape, dscale, MultivariateDiracPM([delta0[1]]))
    p1 = WeibullPHPrior(dshape, dscale, MultivariateDiracPM([delta0[2]]))

    m = WeibullPHModel(WeibullPHParam(1.0, log(2)/5, [delta]),
                        WeibullPHFixedParam(max_n, tmax, emax),
                        (x,y) -> gen_data(x, y, WeibullPHData))

    m, p, p0, p1
end


function getBiPHWeibullModel(max_n, delta0=[log(1.0), log(0.6)], delta=(log(0.8), log(0.6)); gen_data=ph_biweibull_1, tmax=Inf, emax=Inf)##36.0, emax=48.0)
    dshape = Uniform(0.25, 10)

    p = BiPHWeibullPrior((dshape, dshape), (Gamma(1,1), Gamma(1,1)), MvNormal(10*diagm(0 => ones(Float64, 2))), Uniform(0.1, 0.9))
    p0 = BiPHWeibullPrior((dshape, dshape), (Gamma(1,1), Gamma(1,1)), MultivariateDiracPM([0.0, 0.0]), Uniform(0.1, 0.9))
    p1 = BiPHWeibullPrior((dshape, dshape), (Gamma(1,1), Gamma(1,1)), MultivariateDiracPM([log(0.8), log(0.6)]), Uniform(0.1, 0.9))

    m = Model(BiPHWeibullParam((1.0, 1.0), (log(2)/5, log(2)/5), delta, 1/3),
            BiPHWeibullFixedParam(max_n, tmax, emax), (x,y) -> gen_data(x, y, BiPHWeibullData))

    m, p, p0, p1
end

function getBiPHSplineModel(max_n, delta0=[log(1.0), log(0.6)], delta=(log(0.8), log(0.6)); gen_data=ph_billog_1, tmax=36.0, emax=Inf)

    sp = SplineHazard.Prior(Poisson(4), InverseGamma(0.01, 0.01), (theta, v) -> sum(theta.^2)/(2*v), 4, 100,
                                       Vector(0.0:tmax/501:tmax)[2:(end-1)], (0.0, tmax))

    p = BiPHSplinePrior((sp, sp), MvNormal(10*diagm(0 => ones(Float64, 2))), Uniform(0.1, 0.9))
    ##p0 = BiPHSpline.Prior((sp, sp), Dirac.MultivariateDiracPM([delta0[1], delta0[1]]), Uniform(0.1, 0.9))
    ##p1 = BiPHSpline.Prior((sp, sp), Dirac.MultivariateDiracPM([delta0[2], delta0[2]]), Uniform(0.1, 0.9))

    p0 = BiPHSplinePrior((sp, sp), MultivariateDiracPM([0.0, 0.0]), Uniform(0.1, 0.9))
    p1 = BiPHSplinePrior((sp, sp), MultivariateDiracPM([log(0.8), log(0.6)]), Uniform(0.1, 0.9))

    m = BiPHSplineModel(BiPHSplineParam((SplineHazard.Spline.const_spline(1.0, (0.0, tmax)),
                                        SplineHazard.Spline.const_spline(1.0, (0.0, tmax))),
                                                    delta, 1/3),
                                   BiPHSplineFixedParam(max_n, tmax, emax), (x,y) -> gen_data(x, y, BiPHSplineData))

    m, p, p0, p1
end

struct SimResult
    df::DataFrame
    nev::Float64
    asn::Float64
    ndistr::Vector{Float64}
    power_Sc::Float64
    power_S::Float64
    power_both::Float64
    power_any::Float64
    cov1::Float64
    cov2::Float64
    delta1::Float64
    delta2::Float64
end

function melt(x::Vector{Vector{SimResult}})
    R = length(x[1])
    res = Vector{Vector{SimResult}}(undef, R)

    for r in 1:R
        res[r] = [xx[r] for xx in x]
    end

    res
end

collapse(x::DataFrame) = [reduce(g) for g in groupby(x, :rule)]

function reduce(x::SubDataFrame)
    df = by(x, :ia) do zz
        [mean(zz[:nevents]) mean(zz[:n]) mean(zz[:time]) mean((zz[:rejectS] .== 1) .& (zz[:rejectSc] .== 1)) mean((zz[:rejectSc] .== 1) .& (zz[:rejectS] .== 0)) mean((zz[:rejectSc] .== 0) .& (zz[:rejectS] .== 1)) mean(zz[:beta1]) mean(zz[:beta2]) mean(zz[:sd1]) mean(zz[:sd2]) mean(zz[:pd1]) mean(zz[:pd2]) mean(zz[:cilo1]) mean(zz[:cihi1]) mean(zz[:cilo2]) mean(zz[:cihi2]) mean(zz[:cov1]) mean(zz[:cov2]) mean(zz[:u1]) mean(zz[:u2])]
    end

    names!(df, [:ia, :nevents, :n, :time, :reject_both, :reject_Sc_only, :reject_S_only, :beta1, :beta2, :sd1, :sd2, :pd1, :pd2, :cilo1, :cihi1, :cilo2, :cihi2, :cov1, :cov2, :u1, :u2])
    df.rule = repeat([x.rule[1]], inner=nrow(df))

    aux = by(x, :trial_id) do z
        [maximum(z.nevents) maximum(z.n) z.cov1[end] z.cov2[end] any(z.rejectSc .== 1) any(z.rejectS .== 1) (any(z.rejectSc .== 1) & any(z.rejectS .== 1)) (any(z.rejectSc .== 1) | any(z.rejectS .== 1))]
    end
    names!(aux, [:trial_id, :nevents, :n, :cov1, :cov2, :power_Sc, :power_S, :power_both, :power_any])

    SimResult(df, mean(aux.nevents), mean(aux.n), quantile(aux.n), mean(aux.power_Sc), mean(aux.power_S), mean(aux.power_both),
              mean(aux.power_any), mean(aux.cov1), mean(aux.cov2), x[:delta1][1], x[:delta2][1])
end

function summary(rule::String, x::Vector{SimResult})
    asn = [xx.asn for xx in x]
    power = [xx.power for xx in x]
    delta = [xx.delta for xx in x]
    DataFrame(rule=repeat([rule], inner=length(delta)), delta=delta, asn=asn, power=power)
end

summary(x::Dict{String, Vector{SimResult}}) = vcat([summary(v[1], v[2]) for v in x]...)

function get_fsim(B::Int, getmodel::Function, nevents=[[100, 50], [200, 100]]; max_n=400, M=5000,
                  delta0=[log(1.0), log(0.6)], delta=(log(0.8), log(0.6)), tau=0.0,
                  gen_data=ph_biweibull_1, warmup=Int(floor(M/2)), type1=0.05, type2=0.2)

    m, p, p0, p1 = getmodel(max_n, delta0, delta; gen_data=gen_data)
    tpar = GSDTrialParam(2, M, warmup, nevents, max_n, tau, DataFrame(a1=1.0, a2=1.0, b1=1.0, b2=1.0), p)
    get_initial_bounds(B, type1, type2, m, (p, p0, p1), tpar; fsim_only=true)
end

function max_power_fsim(fs)
    c = quantile(maximum(fs[2].post_pd[:,2,:], dims=2)[:,1], 0.95)
    mean(maximum(fs[3].post_pd[:,2,:], dims=2)[:,1] .> c)
end

function get_bounds(fsim, type1=0.05, type2=0.2; ngrid=10, h=0.001, fun=find_gsd_bounds, single_stage=false)
    objs = Dict("gsd" => gsd_utility, "sponsor" => sponsor_utility, "public" => public_utility)

    opt_bounds = @distributed vcat for obj in collect(values(objs))
        fun(type1, type2, fsim[1], fsim[2], fsim[3]; utility=obj, ngrid=ngrid, h=h, single_stage=single_stage)
    end

    DataFrame(rule=["OPT_$(k)" for k in keys(objs)], obj=map(x -> x[2][1], opt_bounds),
              a1=map(x -> x[1][1,1], opt_bounds), a2=map(x -> x[1][2,1], opt_bounds),
              b1=map(x -> x[1][1,2], opt_bounds), b2=map(x -> x[1][2,2], opt_bounds),
              t1=map(x -> x[3], opt_bounds), t2=map(x -> x[4], opt_bounds),
              t1_Sc=map(x -> x[5], opt_bounds), t2_Sc=map(x -> x[6], opt_bounds),
              t1_S=map(x -> x[7], opt_bounds), t2_S=map(x -> x[8], opt_bounds))
end

## run GSD simulation for each rule (each row of bounds DataFrame)
function gsd_sim(iter, bounds::DataFrame;
                 max_d=[[100, 50], [200, 100]], max_n=400, M=5000, delta0=[log(1.0), log(0.6)], delta=(0.0, 0.0),
                 model_fun=getBiPHWeibullModel, gen_data=ph_biweibull_1, warmup=Int(floor(M/2)), tau=0.0)

    rules = unique(bounds[:rule])
    res = Vector{Vector{SimResult}}(undef, 0)

    m, p, p0, p1 = model_fun(max_n, delta0, delta; gen_data=gen_data)
    tpar = GSDTrialParam(2, M, warmup, max_d, max_n, tau, bounds, p)
    tl = GSDTrial(m, tpar)

    trials = @distributed vcat for i in 1:iter
        df = run_trial(tl, i)
        df[:delta1] = delta[1]
        df[:delta2] = delta[2]
        df
    end

    push!(res, collapse(trials))
    dd = Dict(zip(rules, melt(res)))
    dd
end

function weibull_aed_sim(iter, bounds_df::DataFrameRow; delta=(log(1.0), log(0.6)), h=[(0.00025, true), (0.000250001, false)], utility=public_utility, B=10000, M=20000)
    bounds = [bounds_df.a1 bounds_df.b1; bounds_df.a2 bounds_df.b2]
    s1m, s1p, s1p0, s1p1 = getBiPHWeibullModel(400, [log(1.0), log(0.6)], delta)

    s2m_x, s2p_x, s2p0_x, s2p1_x = getWeibullPHModel(400, [log(1.0), log(0.8)], delta[2]; gen_data=ph_exp_1)
    s2m_y, s2p_y, s2p0_y, s2p1_y = getWeibullPHModel(400, [log(1.0), log(0.6)], delta[2]; gen_data=ph_exp_1)

    s2_dbl_m = Model(DoubleWeibullParam(s2m_x.param, s2m_y.param),
                    DoubleWeibullFixedParam(s2m_x.fparam, s2m_y.fparam),
                    s2m_x.generate_data)

    s2_dbl_p = DoubleWeibullPrior(s2p_x, s2p_y)
    s2_dbl_p0 = DoubleWeibullPrior(s2p0_x, s2p0_y)
    s2_dbl_p1 = DoubleWeibullPrior(s2p1_x, s2p1_y)

    s1tpar = AEDTrialParam(2, M, B, [[100, 50], [200, 100]], 0.0, bounds; h=h, utility=utility)
    s2tpar = AEDTrialParam(1, M, B, [[200, 100]], 0.0, bounds; h=h, utility=utility)

    t = AEDTrial(B,
                  s1m, Dict("full" => s1m, "double" => s2_dbl_m),
                  (s1p, s1p0, s1p1), Dict("full" => (s1p, s1p0, s1p1), "double" => (s2_dbl_p, s2_dbl_p0, s2_dbl_p1)),
                  s1tpar, Dict("full" => s2tpar, "double" => s2tpar))

    ##run_with_progress(iter, vcat, it -> AED.run_trial(t, it))
    @sync @distributed vcat for it in 1:iter
        @info "Iteration $(it)/$(iter)"
        run_trial(t, it)
    end
end
