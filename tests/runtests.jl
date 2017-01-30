
using StochDynamicProgramming, Clp, ModelPredictiveControl
using Base.Test


@testset "MPC" begin
    ######## Optimization parameters  ########
    # choose the LP solver used.
    const SOLVER = ClpSolver() 			   # require "using Clp"
    #const SOLVER = CplexSolver(CPX_PARAM_SIMDISPLAY=0) # require "using CPLEX"


    ######## Stochastic Model  Parameters  ########
    const N_STAGES = 6              # number of stages of the SP problem
    const COSTS = [sin(3*t)-1 for t in 1:N_STAGES]
    #const COSTS = rand(N_STAGES)    # randomly generating deterministic costs

    const CONTROL_MAX = 0.5         # bounds on the control
    const CONTROL_MIN = 0

    const XI_MAX = 0.3              # bounds on the noise
    const XI_MIN = 0
    const N_XI = 10                 # discretization of the noise

    const S0 = 0.5                  # initial stock

    # create law of noises
    proba = 1/N_XI*ones(N_XI) # uniform probabilities
    xi_support = collect(linspace(XI_MIN,XI_MAX,N_XI))
    xi_law = NoiseLaw(xi_support, proba)
    xi_laws = NoiseLaw[xi_law for t in 1:N_STAGES-1]

    # Define dynamic of the stock:
    function dynamic(t, x, u, xi)
        return [x[1] + u[1] - xi[1]]
    end

    # Define cost corresponding to each timestep:
    function cost_t(t, x, u, w)
        return COSTS[t] * u[1]
    end

    ######## Setting up the SPmodel
    s_bounds = [(0, 2)] 			# bounds on the state
    u_bounds = [(CONTROL_MIN, CONTROL_MAX)] # bounds on controls
    spmodel = LinearSPModel(N_STAGES, u_bounds, [S0], cost_t, dynamic, xi_laws)

    solver = MPCSolver()
    @test isa(solver, MPCSolver)

    # generate assessment scenarios:
    scenarios = StochDynamicProgramming.simulate_scenarios(spmodel.noises, 1)
    @test isa(scenarios, Array{Float64, 3})

    function oraclegen(t)
        function oracle(s)
            return [(XI_MAX - XI_MIN)/2]
        end
        return oracle
    end

    c, x, u = simulation(spmodel, solver,
                        scenarios,
                        oraclegen,
                        realfinalcost=x->0)

    @test isa(c, Array{Float64, 1})
    @test isa(x, Array{Float64, 3})
    @test isa(u, Array{Float64, 3})
end
