

type MPCSolver
    solver::MathProgBase.AbstractMathProgSolver
    horizon::Int64
    MILP::Bool

    function MPCSolver(;
                    solver=MathProgBase.defaultLPsolver,
                    horizon=-1,
                    relaxation=false
                    )
        return new(solver, horizon, ~relaxation)
    end
end


"""Build JuMP Model corresponding to MPC problem at time t.

# Arguments
* `model::LinearLinearSPModel`
* `t::Int`
    Current time
* `oracle`
    Forecast to consider
* `x0`
    State of the system at time `t`
* `Tf::Int`
    Final time

# Output
* `m::JuMP.Model`

"""
function build(model::LinearSPModel, mpcsolver::MPCSolver,
                    t::Int, oracle::Function, horizon=-1)
    # number of stages in stochastic problem:
    Tf = model.stageNumber

    if horizon < 0
        ntime = Tf - t + 1
    else
        ntime = horizon
    end
    nx = model.dimStates
    nu = model.dimControls

    m = Model(solver=mpcsolver.solver)

    # take into account binary constraints in MPC:
    if model.IS_SMIP
        controlcat = Array{Symbol, 2}(nu, ntime)
        controlcat[:, :] = :Cont
        controlcat[1, :] = :Bin
        @variable(m,  model.ulim[i][1] <= u[i=1:nu, j=1:ntime-1] <=  model.ulim[i][2],
                  category=controlcat[i, j])
    else
        @variable(m,  model.ulim[i][1] <= u[i=1:nu, j=1:ntime-1] <=  model.ulim[i][2])
    end
    # Define constraints other variables:
    @variable(m,  model.xlim[i][1] <= x[i=1:nx, j=1:ntime] <= model.xlim[i][2])

    @variable(m, w0[1:model.dimNoises])
    m.ext[:noise] = @constraint(m, w0 .== oracle(t))
    cost0 = model.costFunctions(t, x[:, 1], u[:, 1], w0)
    costs = [model.costFunctions(t+j-1, x[:, j], u[:, j], oracle(j+t-1)) for j=2:ntime-1]

    # Set optimization objective:
    @objective(m, Min, cost0 + sum(costs[i] for i = 1:ntime-2))

    @constraint(m, x[:, 2] .== model.dynamics(t, x[:, 1], u[:, 1], w0))
    for j in 2:(ntime-1)
        # Dynamic constraint corresponding to system's state function:
        @constraint(m, x[:, j+1] .== model.dynamics(j+t-1, x[:, j], u[:, j], oracle(t+j-1)))
    end

    # Add initial constraints:
    m.ext[:cons] = @constraint(m, init_cons, x[:, 1] .== 0)

    return m
end


"""Solve MPC Problem at time t and position x0.

# Arguments
- `mpcprob::JuMP.Model`
    MPC optimization problem
- `x0::Array`
    Current position

# Outputs
- `u::Array`
    Optimal control found by MPC
"""
function solvempc(mpcprob::JuMP.Model, x0::Array{Float64, 1})
    u = getvariable(mpcprob, :u)
    for i in 1:length(x0)
        JuMP.setRHS(mpcprob.ext[:cons][i], x0[i])
    end

    # Solve MPC problem via JuMP:
    st = solve(mpcprob)

    if st == :Optimal
        return collect(getvalue(u)[:, 1])
    else
        println(mpcprob)
        return zeros(length(u[:, 1]))
    end
end


""" Compute MPC optimal control other given scenario.

# Arguments
* `model::LinearSPModel`
* `mpc::MPC`
* `x0`
    Initial state
* `scenario`
    Scenario of perturbations to consider
* `forecast`::Array
    Forecast provided to MPC
* `real_dynamic::Function`
* `realcost::Function`
* `info::Int`
    Whether to update forecast at time `t` or not

# Returns
* `costs`
* `stocks`
* `controls`

"""
function simulation(model::LinearSPModel,
                    mpcsolver::MPCSolver,
                    scenarios::Array,
                    oraclegen::Function;
                    real_dynamic=nothing,
                    real_cost=nothing,
                    realfinalcost=nothing,
                    verbose=0)

    if isa(real_dynamic, Void)
        real_dynamic = model.dynamics
    end
    if isa(real_cost, Void)
        real_cost = model.costFunctions
    end
    if isa(realfinalcost, Void)
        realfinalcost = model.finalCost
    end

    # Get number of timesteps:
    T = model.stageNumber
    @assert T-1 == size(scenarios, 1)

    nb_simulations = size(scenarios, 2)

    # Arrays to store optimal trajectories
    stocks = zeros(T, nb_simulations, model.dimStates)
    controls = zeros(T, nb_simulations, model.dimControls)
    costs = zeros(nb_simulations)

    # Set first value of stocks equal to x0:
    for i in 1:nb_simulations
        stocks[1, i, :] = model.initialState
    end

    p = Progress(T-1, 1)
    for t=1:T-1
        # update oracle at time t:
        oracle = oraclegen(t)
        mpc_prob = build(model, mpcsolver, t, oracle)
        for k in 1:nb_simulations
            # get previous state:
            xt = stocks[t, k, :]

            # find optimal control with MPC:
            mpccontrol = solvempc(mpc_prob, xt)

            # get current perturbation
            ξ = vec(scenarios[t, k, :])

            costs[k] += real_cost(t, xt, mpccontrol, ξ)
            xf = real_dynamic(t, xt, mpccontrol, ξ)

            # store results in array
            stocks[t+1, k, :] = xf
            controls[t, k, :] = mpccontrol
        end
        # update progress bar
        next!(p)
    end

    # Get final cost
    for k = 1:nb_simulations
        costs[k] += realfinalcost(stocks[end, k, :])
    end
    return costs, stocks, controls
end

