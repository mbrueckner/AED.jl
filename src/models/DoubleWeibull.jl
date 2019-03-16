## dummy model to pack to WeibullPH models together

struct DoubleWeibullData <: AbstractData
    x::WeibullPHData
    y::WeibullPHData
end

nevents(data::DoubleWeibullData) = sum(data.x.status) + sum(data.y.status)

DoubleWeibullData() = DoubleWeibullData(WeibullPHData(), WeibullPHData())

mutable struct DoubleWeibullParam <: AbstractParam
    x::WeibullPHParam
    y::WeibullPHParam
end

struct DoubleWeibullFixedParam <: AbstractFixedParam
    x::WeibullPHFixedParam
    y::WeibullPHFixedParam
end

struct DoubleWeibullPrior <: AbstractPrior
    x::WeibullPHPrior
    y::WeibullPHPrior
end

DoubleWeibullPrior() = DoubleWeibullPrior(WeibullPHPrior(), WeibullPHPrior())

function DoubleWeibullModel()
    m1 = WeibullPHModel()
    m2 = WeibullPHModel()
    Model(DoubleWeibullParam(m1.param, m2.param), DoubleWeibullFixedParam(m1.fparam, m2.fparam), generate_data)
end

DoubleWeibullModel(param, fparam, gen_data=generate_data) = Model(param, fparam, gen_data)

struct DoubleWeibullFit <: AbstractFit
    x::WeibullPHFit
    y::WeibullPHFit
end

slice(i::Int, x::DoubleWeibullFit) = Param(slice(x.x), slice(x.y))

function sample_prior(B::Int, prior::DoubleWeibullPrior, fpar)
    par1 = sample_prior(B, prior.x, fpar.x)
    par2 = sample_prior(B, prior.y, fpar.y)

    par = Vector{Param}(undef, B)
    for b in 1:B
        par[b] = Param(par1[b], par2[b])
    end

    par
end

## return prior distribution (before sampling) or posterior sample (after sampling)
## of treatment effect parameter
function get_outcome(x::DoubleWeibullFit)
    if size(x.beta, 1) == 0
        [x.prior.x.beta, x.prior.y.beta]
    else
        hcat(x.x.beta, x.y.beta)
    end
end

generate_data(par, fpar) = predict_data(par, fpar, DoubleWeibullData(), 0.0)

function predict_data(par::DoubleWeibullParam, fpar::DoubleWeibullFixedParam, data0::DoubleWeibullData, caltime::Float64, nevents=NaN)
    DoubleWeibullData(predict_data(par.x, fpar.x, data0.x, caltime),
                        predict_data(par.y, fpar.y, data0.y, caltime))
end

function select_data(t::Float64, data::DoubleWeibullData; entry_stop=t)
    DoubleWeibullData(select_data(t, data.x; entry_stop=entry_stop),
         select_data(t, data.y; entry_stop=entry_stop))
end

function laplace_approx(data::DoubleWeibullData, param::DoubleWeibullParam, prior::DoubleWeibullPrior)
    vcat(laplace_approx(data.x, param.x, prior.x),
         laplace_approx(data.y, param.y, prior.y))
end

function get_approx(param::DoubleWeibullParam, new_data::DoubleWeibullData, prior::DoubleWeibullPrior)
    laplace_approx(new_data, param, prior)
end

function fit(M::Int, data::DoubleWeibullData; prior::DoubleWeibullPrior=DoubleWeibullPrior(), warmup::Int=Int(floor(M/2)))
    DoubleWeibullFit(fit(M, data.x; prior=prior.x, warmup=warmup), fit(M, data.y, prior=prior.y, warmup=warmup))
end

summary(x::DoubleWeibullFit) = summary(x.x), summary(x.y)

function doubleweibull_test_laplace_approx()
    m = DoubleWeibullModel()
    data = m.generate_data(m.param, m.fparam)

    la = laplace_approx(data, m.param, DoubleWeibullPrior())
    x = fit(10000, data; warmup=5000)

    print(summary(x))
    x, la
end
