struct GSDTrialParam
    max_ia::Int
    M::Int
    warmup::Int
    nevents::Vector{Vector{Int}}
    max_n::Int
    delta::Float64
    bounds::DataFrame
    prior::AbstractPrior

    function GSDTrialParam(max_ia::Int, M::Int, warmup::Int, nevents::Vector{Vector{Int}}, max_n::Int, delta::Float64, bounds::DataFrame, p::AbstractPrior)
        new(max_ia, M, warmup, nevents, max_n, delta, bounds, p)
    end
end

struct GSDTrial
    model::Model
    param::GSDTrialParam
end

check_bounds(pd1, pd2, u1, u2) = (pd1 > u1) || (pd2 > u2)

## simulate a single one-sided k-stage group-sequential design
function run_trial(trial::GSDTrial, id::Int)
    max_ia = trial.param.max_ia
    M = trial.param.M
    warmup = trial.param.warmup

    model = trial.model

    ## initial data
    data = model.generate_data(model.param, model.fparam) ## BiPHWeibull.generate_data(model.param, model.fparam) ##

    bounds = trial.param.bounds
    z = groupby(bounds, :rule)

    R = length(z)
    stopped = zeros(Bool, R)

    res = DataFrame(trial_id=Int64[], rule=String[], ia=Int64[], time=Float64[], rejectSc=Int64[], rejectS=Int64[],
                    beta1=Float64[], beta2=Float64[],
                    sd1=Float64[], sd2=Float64[], pd1=Float64[], pd2=Float64[], cilo1=Float64[], cihi1=Float64[],
                    cilo2=Float64[], cihi2=Float64[], cov1=Bool[], cov2=Bool[], u1=Float64[], u2=Float64[], n=Int64[], nevents=Int64[])

    true_beta1 = model.param.beta[1]
    true_beta2 = model.param.beta[2]

    for j in 1:max_ia
        ia_time, ia_data = select_ia_data(trial.param.nevents[j], data)

        n = length(ia_data.time)

        x = fit(M, ia_data; warmup=warmup, prior=trial.param.prior)

        beta = get_outcome(x)
        beta1 = beta[:,1]
        beta2 = beta[:,2]

        pd1 = mean(beta1 .<= trial.param.delta)
        pd2 = mean(beta2 .<= trial.param.delta)

        pd1_ci_lo = quantile(beta1, 0.025)
        pd1_ci_hi = quantile(beta1, 0.975)
        pd2_ci_lo = quantile(beta2, 0.025)
        pd2_ci_hi = quantile(beta2, 0.975)

        for r in 1:R
            if !stopped[r]
                u1 = z[r][[:a1, :a2]][j][1]
                u2 = z[r][[:b1, :b2]][j][1]

                rejectSc = pd1 > u1
                rejectS = pd2 > u2
                stopped[r] = rejectSc || rejectS

                push!(res, [id, "$(z[r][:rule][1])", j, ia_time, rejectSc, rejectS,
                            mean(beta1), mean(beta2), sqrt(var(beta1)), sqrt(var(beta2)),
                            pd1, pd2, pd1_ci_lo, pd1_ci_hi, pd2_ci_lo, pd2_ci_hi,
                            (pd1_ci_lo <= true_beta1) & (true_beta1 <= pd1_ci_hi),
                            (pd2_ci_lo <= true_beta2) & (true_beta2 <= pd2_ci_hi),
                            u1, u2, n, sum(ia_data.status)])
            end
        end

        if all(stopped)
            break
        end
    end

    res
end

## find optimal bounds maximizing expected utility for a one-sided two-stage group-sequential design given set of forward simulations
function find_gsd_bounds(type1, type2, fsim, fsim0, fsim1; single_stage=false, ngrid=10, h=0.001, utility=gsd_utility)
    a1 = quantile(fsim0.post_pd[:,1,1], 1-type1)
    a2 = quantile(fsim0.post_pd[:,2,1], 1-type1)##*(1-type1))
    b1 = quantile(fsim0.post_pd[:,1,2], 1-type1)
    b2 = quantile(fsim0.post_pd[:,2,2], 1-type1)##*(1-type1))

    @show a1, a2, b1, b2

    if single_stage
        gamma, best_obj = SA.grid_search(repeat([ngrid], inner=2), (a2, b2), (1.0, 1.0),
                                      x -> objective([1,2], [1.0 1.0; x[1] x[2]], type1, type2, fsim, fsim0, fsim1;
                                                         h=h, utility=utility))
        u = [1.0 1.0; gamma[1] gamma[2]]
    else
        gamma, best_obj = SA.grid_search(repeat([ngrid], inner=4), (a1, a2, b1, b2), (1.0, 1.0, 1.0, 1.0),
                                      x -> objective([1,2], reshape(collect(x), 2, 2), type1, type2, fsim, fsim0, fsim1;
                                                         h=h, utility=utility))
        u = reshape(collect(gamma), 2, 2)
    end

    real_t1 = alpha(fsim0.post_pd, u)
    real_t2 = 1-alpha(fsim1.post_pd, u)

    real_t1_Sc = alpha(fsim0.post_pd[:,:,1:1], u[:,1:1])
    real_t2_Sc = 1-alpha(fsim1.post_pd[:,:,1:1], u[:,1:1])

    real_t1_S = alpha(fsim0.post_pd[:,:,2:2], u[:,2:2])
    real_t2_S = 1-alpha(fsim1.post_pd[:,:,2:2], u[:,2:2])

    u, best_obj, real_t1, real_t2, real_t1_Sc, real_t2_Sc, real_t1_S, real_t2_S
end

## forward simulate one-sided two-stage group-sequential design and find optimal bounds
function get_initial_bounds(B::Int, type1, type2, m::Model, priors, tpar::GSDTrialParam; single_stage=false, h=0.001, fsim_only=false, ngrid=10)
    data = m.generate_data(m.param, m.fparam)
    data0 = select_data(0.0, data)
    x = fit(0, data0; prior=priors[1])
    x0 = fit(0, data0; prior=priors[2])
    x1 = fit(0, data0; prior=priors[3])

    J = tpar.max_ia
    ifrac = map(sum, tpar.nevents) ./ sum(tpar.nevents[end])

    fsim, fsim0, fsim1 = fsim_model(2, B, tpar.M, tpar.warmup, m, data0, priors, tpar.delta, tpar.nevents)

    if fsim_only
        fsim, fsim0, fsim1
    else
        find_gsd_bounds(type1, type2, fsim, fsim0, fsim1; single_stage=single_stage, h=h, ngrid=ngrid)
    end
end
