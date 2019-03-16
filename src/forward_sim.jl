import Base.convert
import Base.size

struct ForwardSim
    post_mean::Array{Float64,3}
    post_sd::Array{Float64,3}
    post_pd::Array{Float64,3}
    n::Matrix{Int}
    param::Matrix{Float64}
end

function get_interim_time(d::Int, data::AbstractData)
    Y = data.entry .+ data.time
    ev = Y[data.status]

    success = true

    if d > length(ev)
        @debug length(data.time), length(ev), d
        @debug "Requested number of events exceeds total number of events!"
        success = false
    end

    if length(ev) == 0
        @debug "No events"
        maximum(Y), false
    else
        events = sort(ev) ## ordered event times
        events[min(d, length(events))], success
    end
end

function append_data(x::T, y::T; add_ids=false) where T <: Union{BiPHWeibullData, BiPHSplineData}
    if add_ids
        id = vcat(x.id, maximum(x.id) .+ y.id)
    else
        id = vcat(x.id, y.id)
    end
    T(id, vcat(x.entry, y.entry), vcat(x.time, y.time), vcat(x.status, y.status), vcat(x.b, y.b), vcat(x.z, y.z))
end

function append_data(x::T, y::T; add_ids=false) where T <: WeibullPHData
    if add_ids
        id = vcat(x.id, maximum(x.id) .+ y.id)
    else
        id = vcat(x.id, y.id)
    end
    T(id, vcat(x.entry, y.entry), vcat(x.time, y.time), vcat(x.status, y.status), vcat(x.z, y.z))
end

function slice_data(x::T, sel::BitArray{1}) where T <: Union{BiPHWeibullData, BiPHSplineData}
    T(x.id[sel], x.entry[sel], x.time[sel], x.status[sel], x.b[sel], x.z[sel])
end

function select_ia_data(nevents, max_data::WeibullPHData; n_max::Int=size(max_data))
    at, success = get_interim_time(nevents, max_data)
    at_max = sort(max_data.entry)[min(length(max_data.entry),n_max)]
    at = min(at, at_max)
    at, select_data(at, max_data)
end

function select_ia_data(nevents, max_data::DoubleWeibullData; n_max::Int=size(max_data))
    at1, success = get_interim_time(sum(nevents), max_data.x)
    at2, success = get_interim_time(sum(nevents), max_data.y)

    et1 = sort(max_data.x.entry)[min(length(max_data.x.entry),n_max)]
    et2 = sort(max_data.y.entry)[min(length(max_data.y.entry),n_max)]

    max(at1, at2), DoubleWeibullData(select_data(at1, max_data.x; entry_max=min(et1, at1)),
                                      select_data(at2, max_data.y; entry_max=min(et2, at2)))
end

function select_ia_data(nevents, max_data::Union{BiPHWeibullData, BiPHSplineData}; n_max::Int=size(max_data))
    max_data_Sc = slice_data(max_data, max_data.b .== 1)
    max_data_S = slice_data(max_data, max_data.b .== 2)
    select_ia_data(nevents, max_data_Sc, max_data_S; n_max=n_max)
end

function select_ia_data(nevents, max_data_Sc, max_data_S; n_max=size(max_data_Sc)+size(max_data_S))
    at_Sc, s1 = get_interim_time(nevents[1], max_data_Sc)
    at_S, s2 = get_interim_time(nevents[2], max_data_S)

    if !s1 || !s2
        @debug length(max_data_Sc.time), length(max_data_S.time)
        @debug nevents, sum(max_data_Sc.status), sum(max_data_S.status)
        @debug at_Sc, at_S
    end

    at = max(at_Sc, at_S)

    new_data_Sc = select_data(at, max_data_Sc; entry_stop=at_Sc)
    new_data_S = select_data(at, max_data_S; entry_stop=at_S)
    new_data = append_data(new_data_Sc, new_data_S)

    if size(new_data) > n_max
        e_max = sort(new_data.entry)[min(length(new_data.entry),n_max)]
        new_data = select_data(new_data, e_max)
    end

    at, new_data
end

function extract_subpop(bm_pos::Bool, x::BiPHWeibullData)
    b = bm_pos ? 2 : 1
    sel = (x.b .== b)
    WeibullPH.Data(x.id[sel], x.entry[sel], x.time[sel], x.status[sel], x.z[sel,:])
end

get_par(param::Union{BiPHWeibullParam, BiPHSplineParam}) = vcat(collect(param.beta), param.prev)
get_par(param::DoubleWeibullParam) = vcat(param.x.beta, param.y.beta, 0.0)


convert(::Type{DoubleWeibullParam}, x::BiPHWeibullParam) = DoubleWeibullParam(WeibullPHParam(x.shape[1], x.scale[1], [x.beta[1]]),
                                                                              WeibullPHParam(x.shape[2], x.scale[2], [x.beta[2]]))

convert(::Type{DoubleWeibullData}, x::BiPHWeibullData) = DoubleWeibullData(extract_subpop(false, x), extract_subpop(true, x))


size(data::Union{BiPHWeibullData, BiPHSplineData}) = length(data.time)
size(data::WeibullPHData) = length(data.time)
size(data::DoubleWeibullData) = size(data.x) + size(data.y)

nparams(pr::Union{BiPHWeibullPrior, BiPHSplinePrior}) = length(pr.beta)
nparams(pr::DoubleWeibullPrior) = length(pr.x.beta) + length(pr.y.beta)

get_model_data(::Type{BiPHWeibullParam}, data::BiPHWeibullData) = data
get_model_data(::Type{BiPHSplineParam}, data::BiPHSplineData) = data
get_model_data(::Type{DoubleWeibullParam}, data::BiPHWeibullData) = convert(DoubleWeibullData, data)

## forward simulation of future IAs
function forward_simulate(J::Int, ia_time::Float64, delta, max_ev, data::D,
    m::Model{T,S}, x::F, prior0::P; B=10000) where {D <: AbstractData, F <: AbstractFit, P <: AbstractPrior, T <: AbstractParam, S <: AbstractFixedParam}

    B0 = size(x.beta, 1)
    K = nparams(prior0)

    n = SharedArray{Int,2}((B, J))
    post_mean = SharedArray{Float64,3}((B, J, K))
    post_sd = SharedArray{Float64,3}((B, J, K))
    post_pd = SharedArray{Float64,3}((B, J, K))
    par = SharedArray{Float64,2}((B,K+1))

    nmiss = SharedArray{Int,1}((J))
    nmiss .= 0

    @sync @distributed for b in 1:B
        param = slice(sample(1:B0), x)

        ## convert BiPHWeibullParam to DoubleWeibullParam
        model_param = convert(T, param)
        model_data = get_model_data(T, data)
        max_data = predict_data(model_param, m.fparam, model_data, ia_time, max_ev[end])

        for js in 1:J
            at, new_data = select_ia_data(max_ev[js], max_data)
            par[b,:] = get_par(param)

            if nevents(new_data) < sum(max_ev[js])
                @debug nevents(new_data), max_ev[js]
                nmiss[js] += 1
            end

            beta_sample = get_approx(model_param, new_data, prior0)
            post_mean[b, js, :] = mean.(beta_sample)
            post_sd[b, js, :] = sqrt.(var.(beta_sample))
            if K == 2
                post_pd[b, js, :] = [cdf(beta_sample[1], delta), cdf(beta_sample[2], delta)]
            else
                post_pd[b, js, :] = [cdf(beta_sample[1], delta)]
            end

            ##end
            n[b,js] = size(new_data)
        end
    end

    @debug nmiss

    ForwardSim(post_mean, post_sd, post_pd, n, par)
end

function fsim_model(K, B, M, warmup, m::Model, data::AbstractData, p::NTuple{3,AbstractPrior}, delta, nevents)
    x = fit(M, data; prior=p[1], warmup=warmup)
    x0 = fit(M, data; prior=p[2], warmup=warmup)
    x1 = fit(M, data; prior=p[3], warmup=warmup)

    fsim = forward_simulate(K, 0.0, delta, nevents, data, m, x, x.prior; B=B)
    fsim0 = forward_simulate(K, 0.0, delta, nevents, data, m, x0, x.prior; B=B)
    fsim1 = forward_simulate(K, 0.0, delta, nevents, data, m, x1, x.prior; B=B)

    fsim, fsim0, fsim1
end
