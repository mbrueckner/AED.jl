## A discrete distribution representing a point mass (DiracPM) or several point masses (MultvariateDiracPM)
## Like Distributions.Categorical but remembers the posiition of the point mass
## This is needed to specify the priors under the null and alternative hypotheses

struct DiracPM <: DiscreteUnivariateDistribution
    x::Float64
end

struct MultivariateDiracPM <: DiscreteMultivariateDistribution
    x::Vector{Float64}
end

import Statistics.var, Statistics.mean, Base.rand, Distributions.cdf, Base.length

mean(d::DiracPM) = d.x
var(d::DiracPM) = 0.0
rand(d::DiracPM) = d.x
rand(d::DiracPM, B::Int) = repeat([d.x], inner=B)
cdf(d::DiracPM, x::Float64) = Float64(d.x <= x)
length(d::DiracPM) = 1

mean(d::MultivariateDiracPM) = d.x
var(d::MultivariateDiracPM) = 0.0
rand(d::MultivariateDiracPM) = d.x
rand(d::MultivariateDiracPM, B::Int) = reshape(repeat(d.x, outer=B), length(d.x), B)
cdf(d::MultivariateDiracPM, x::Vector{Float64}) = Float64(all(d.x .<= x))
length(d::MultivariateDiracPM) = length(d.x)
