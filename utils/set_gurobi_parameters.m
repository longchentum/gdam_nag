%Default parameters when using gurobi as the sub-problem solver
function gurobi_params = set_gurobi_parameters()
  gurobi_params.presolve = 1;
  %All constraints must be satisfied to a tolerance. default: 1e-6
  %gurobi_params.FeasibilityTol=1e-4;
  %The relative difference between the primal and dual objective values
  %default: 1e-8
  %gurobi_params.BarConvTol=1e-9;
  %The improving direction in order for a model to be declared optimal. 
  %default: 1e-6
  %gurobi_params.OptimalityTol=1e-9;
  %gurobi_params.outputflag = 0;
  %Default all threads
  %gurobi_params.threads = 1;
  %In case Q is indefinite
  %gurobi_params.NonConvex = 2;
  %Time limit is sec
  gurobi_params.TimeLimit = 2400;
end
