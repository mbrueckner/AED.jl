struct AEDTrialParam
    max_ia::Int
    M::Int
    warmup::Int
    nevents::Vector{Vector{Int}}
    delta::Float64
    bounds::Matrix{Float64}
    h::Vector{Tuple{Float64,Bool}}
    utility::Function

    function AEDTrialParam(max_ia::Int, M::Int, warmup::Int, d::Vector{Vector{Int}},
                        delta::Float64, bounds::Matrix{Float64};
                        h=[(0.00025, true), (0.00025, false)], utility::Function=sponsor_utility)
        if M <= warmup
            @error "Total number of iterations must be larger than warmup iterations"
        end

        new(max_ia, M, warmup, d, delta, bounds, h, utility)
    end
end

struct AEDTrial
    B::Int
    s1_model::Model
    s2_models::Dict{String,Model} ## bi-model, Sc only, S only

    s1_prior::NTuple{3,AbstractPrior}
    s2_priors::Dict{String,NTuple{3,AbstractPrior}}

    s1_param::AEDTrialParam
    s2_params::Dict{String,AEDTrialParam}
end

function extract_subset(x::BiPHWeibullData, sel::BitArray{1})
    BiPHWeibullData(x.id[sel], x.entry[sel], x.time[sel], x.status[sel], x.b[sel], x.z[sel])
end

function extract_subset(x::WeibullPHData, sel::BitArray{1})
    WeibullPHData(x.id[sel], x.entry[sel], x.time[sel], x.status[sel], x.z[sel,:])
end

function find_bounds(A::Vector{Int}, fsim::NTuple{3,ForwardSim}, cef::Float64, cef_t2::Float64; h=0.001, ngrid=50, utility=gsd_utility)
    if length(A) == 2
        upper, obj = SA.grid_search([ngrid, ngrid], [0.90, 0.90], [1.0, 1.0],
                                    x -> objective(A, reshape(collect(x), 1, 2), cef, 1.0, fsim[1], fsim[2], fsim[3]; h=h, utility=utility)[1])
        collect(upper), objective(A, reshape(collect(upper), 1, 2), cef, cef_t2, fsim[1], fsim[2], fsim[3]; h=h, utility=utility)
    else
        tmp = quantile(fsim[2].post_pd[:,1,A[1]], 1-cef)
        upper = ones(Float64, 1, 2)
        upper[1, A[1]] = tmp
        upper[1,:], objective(A, upper, cef, 1.0, fsim[1], fsim[2], fsim[3]; h=h, utility=utility)
    end
end

function get_selection(tpf, tpd, ia_data, fsim, fsim0, fsim1, fsim_dbl, fsim0_dbl, fsim1_dbl, cef, cef_t2, h, utility; allow_Sc_only=true)
    B = size(fsim.post_pd, 1)
    upperF, objF = find_bounds([1,2], (fsim, fsim0, fsim1), cef, cef_t2; h=h, utility=utility)
    upperSc, objSc = find_bounds([1], (fsim_dbl, fsim0_dbl, fsim1_dbl), cef, cef_t2; h=h, utility=utility)
    upperS, objS = find_bounds([2], (fsim_dbl, fsim0_dbl, fsim1_dbl), cef, cef_t2; h=h, utility=utility)

    n = length(ia_data.time)
    objFut = objective(Vector{Int}(undef, 0), zeros(Float64, 1, 2), zeros(Float64, B, 1, 2), n.*ones(Int, B, 1), fsim.param; h=h)

    A = [Vector{Int}(undef,0), [1,2], [2], [1]]
    obj = [objFut, objF[1], objS[1]]

    if allow_Sc_only
        obj = vcat(obj, objSc[1])
    end

    upper = [[0.0, 0.0], upperF, upperS, upperSc]
    k = findmax(obj)[2]

    @debug "selection", A[k], obj, upper, k, cef, cef_t2, h, allow_Sc_only

    A[k], obj[k], upper[k], h, allow_Sc_only
end

function select_population(B::Int, pd1, x_bi, ia_data, ia_time, m::Dict{String,Model}, p::Dict{String,NTuple{3,Tp}},
                           tpar::Dict{String,AEDTrialParam}, old_fa_bounds; utility=gsd_utility) where Tp <: AbstractPrior
    tpf = tpar["full"]
    tpd = tpar["double"]

    x0_bi = fit(tpf.M, ia_data; prior=p["full"][2], warmup=tpf.warmup)
    x1_bi = fit(tpf.M, ia_data; prior=p["full"][3], warmup=tpf.warmup)

    fsim = forward_simulate(1, ia_time, tpf.delta, tpf.nevents, ia_data, m["full"], x_bi, x_bi.prior; B=B)
    fsim0 = forward_simulate(1, ia_time, tpf.delta, tpf.nevents, ia_data, m["full"], x0_bi, x_bi.prior; B=B)
    fsim1 = forward_simulate(1, ia_time, tpd.delta, tpf.nevents, ia_data, m["full"], x1_bi, x_bi.prior; B=B)

    prior_dbl = p["double"][1]
    fsim_dbl = forward_simulate(1, ia_time, tpd.delta, tpd.nevents, ia_data, m["double"], x_bi, prior_dbl; B=B)
    fsim0_dbl = forward_simulate(1, ia_time, tpd.delta, tpd.nevents, ia_data, m["double"], x0_bi, prior_dbl; B=B)
    fsim1_dbl = forward_simulate(1, ia_time, tpd.delta, tpd.nevents, ia_data, m["double"], x1_bi, prior_dbl; B=B)

    cef = alpha(fsim0.post_pd, old_fa_bounds)
    cef_t2 = 1-alpha(fsim1.post_pd, old_fa_bounds)

    [get_selection(tpf, tpd, ia_data, fsim, fsim0, fsim1, fsim_dbl, fsim0_dbl, fsim1_dbl, cef, cef_t2, hh[1], utility; allow_Sc_only=hh[2]) for hh in tpf.h]
end

check_bounds(pd, bnds) = pd .> bnds

function fit_model(M, warmup, data, prior, delta)
    x = fit(M, data; warmup=warmup, prior=prior)
    beta = get_outcome(x)
    mean(beta .<= delta, dims=1)[1,:], x
end

function split(m::Model{T,S}) where {T <: DoubleWeibullParam, S <: DoubleWeibullFixedParam}
    Model(m.param.x, m.fparam.x, m.generate_data), Model(m.param.y, m.fparam.y, m.generate_data)
end

function stage2(id, A, obj, fa_bnds, h, res, ia_data, s1_data, s2_params, s2_models, s2_priors)
    @info id, A

    ## Stage 2
    if A == [1,2]
        s2p = s2_params["full"]
        s2_model = s2_models["full"]
        s2_prior = s2_priors["full"][1]
        nev2 = s2p.nevents[1]
        subpop = "full"
    else
        s2p = s2_params["double"]
        nev2 = sum(s2p.nevents[1])
        if A == [1]
            s2_model = split(s2_models["double"])[1]
            s2_prior = s2_priors["double"][1].x
            s1_data = extract_subpop(false, s1_data)
            subpop = "bm-neg"
        else
            s2_model = split(s2_models["double"])[2]
            s2_prior = s2_priors["double"][1].y
            s1_data = extract_subpop(true, s1_data)
            subpop = "bm-pos"
        end
    end

    s2_data = s2_model.generate_data(s2_model.param, s2_model.fparam)

    ## all subjects from IA but with post-IA follow-up
    s1_data = extract_subset(s1_data, s1_data.id .<= maximum(ia_data.id))

    ## first stage 2 subject enrolled after last IA subject
    s2_data.entry .+= maximum(s1_data.entry)

    new_data = append_data(s1_data, s2_data; add_ids=true)
    fa_time, fa_data = select_ia_data(nev2, new_data; n_max=400)

    pd = [NaN, NaN]
    tmp, x2 = fit_model(s2p.M, s2p.warmup, fa_data, s2_prior, s2p.delta)
    pd[A] .= tmp

    reject = check_bounds(pd, fa_bnds)

    push!(res, [id, 2, h[1], h[2], length(fa_data.time), fa_time, subpop, reject[1], reject[2],
                Int(any(reject)), pd[1], pd[2], sum(fa_data.status)])
end

function run_trial(trial::AEDTrial, id::Int)
    s1p = trial.s1_param

    ## Stage 1
    s1_model = trial.s1_model
    s1_data = s1_model.generate_data(s1_model.param, s1_model.fparam)

    ## initial 2-stage GSD bounds
    bounds = trial.s1_param.bounds

    ## stage 1 model fit
    ia_time, ia_data = select_ia_data(s1p.nevents[1], s1_data)
    pd, x1 = fit_model(s1p.M, s1p.warmup, ia_data, trial.s1_prior[1], s1p.delta)
    reject = check_bounds(pd, bounds[1,:])

    ## empty DataFrame to hold results
    res = DataFrame(trial_id=Vector{Int}(undef,0), ia=Vector{Int}(undef,0), h=Vector{Float64}(undef,0), allowSc=Vector{Bool}(undef,0),
                    n=Vector{Int}(undef,0),
                    time=Vector{Float64}(undef,0), subpop=Vector{String}(undef,0), rejectSc=Vector{Bool}(undef,0), rejectS=Vector{Bool}(undef,0),
                    decision=Vector{Int}(undef,0), pdSc=Vector{Float64}(undef,0), pdS=Vector{Float64}(undef,0), nevents=Vector{Int}(undef,0))

    ## add stage 1 results to DataFrame for every "h" value
    for h in trial.s1_param.h
        push!(res, [id, 1, h[1], h[2], length(ia_data.time), ia_time, "full", reject[1], reject[2], Int(any(reject)), pd[1], pd[2], sum(ia_data.status)])
    end

    @info id, reject

    if !any(reject)
        ## select population
        sel_vec = select_population(trial.B, pd, x1, ia_data, ia_time, trial.s2_models, trial.s2_priors,
                                    trial.s2_params, reshape(bounds[2,:], 1, 2); utility=s1p.utility)

        for k in 1:length(sel_vec)
            sv = sel_vec[k]
            A, obj, fa_bnds, h, allow_Sc_only = sv[1], sv[2], sv[3], sv[4], sv[5]

            if length(A) == 0
                ## res.subpop[1] = "fut"
                push!(res, [id, 2, h, allow_Sc_only, length(ia_data.time), ia_time, "fut", false, false, 0, NaN, NaN, sum(ia_data.status)])
            else
                stage2(id, A, obj, fa_bnds, (h, allow_Sc_only), res, ia_data, s1_data, trial.s2_params, trial.s2_models, trial.s2_priors)
            end
        end
    end

    res
end

function summary(x::DataFrame)
    res = by(x, [:h, :trial_id]) do zz
        tmp = zz[end,:]
        [tmp.n tmp.time tmp.nevents tmp.subpop (tmp.rejectSc | tmp.rejectS) (tmp.rejectSc & tmp.rejectS) tmp.rejectSc tmp.rejectS tmp.pdSc tmp.pdS]
    end
    names!(res, [:h, :trial_id, :n, :time, :nevents, :subpop, :any, :both, :Sc, :S, :pdSc, :pdS])
    res
end
