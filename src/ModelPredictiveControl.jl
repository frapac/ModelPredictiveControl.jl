# Implementation of Model Predictive Control in Julia

module ModelPredictiveControl

using StochDynamicProgramming, ProgressMeter, JuMP, MathProgBase

export MPCSolver, simulation, solve
include("mpc.jl")

end
