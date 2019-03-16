module AED

using Distributed, SharedArrays, DataFrames, CSV
using Statistics, StatsBase, Distributions
using LinearAlgebra, Optim, Roots, ForwardDiff

using SplineHazard
import SplineHazard.Spline.Dierckx.Spline1D


abstract type AbstractData end
abstract type AbstractParam end
abstract type AbstractFixedParam end
abstract type AbstractFit end
abstract type AbstractPrior end

struct Model{T <: AbstractParam, S <: AbstractFixedParam}
    param::T
    fparam::S
    generate_data::Function
end

include("SA.jl")
include("Dirac.jl")
include("MCMCUtil.jl")

import .SA: grid_search
import .MCMCUtil: MH_single, AM_RR_single, updateMean, updateVar

include("models/BiPHSpline.jl")
include("models/WeibullPH.jl")
include("models/DoubleWeibull.jl")
include("models/BiPHWeibull.jl")

include("forward_sim.jl")
include("objective.jl")

include("gsd_trial.jl")
include("aed_trial.jl")

include("sim.jl")

"""
A Julia package for Bayesian adaptive enrichment designs with subgroup selection
"""
AED

end
