% This is an implementation of the gradient descent akin method (GDAM)
% with Nesterov's Accleration Gradient (NAG) for sparse QPs.
%
% min  1/2 x^T Q x + c^T x, 
% s.t. Aineq x <= bineq
%      Aeq x = beq
%      lb <= x <= ub
%
% The results obtained by GDAM+NAG are compared with the MATLAB quadprog
% Gurobi, and the opensource QP solver OSQP 
%
% Copyright (C) 2022 
% Author: Long Chen,
%
% Reference: L. Chen, W. Chen, and K.-U. Bletzinger, A gradient descent 
% akin method for inequality constrained optimization, arXiv:1902.04040v4
%
%clear;
%function [methods, fvals, iters, runtimes, errors] = convexQP_sparse(model, solvers)
function ResTable = convexQP_sparse(model, solvers)
% Initialization
methods = [];
fvals = [];
iters = [];
runtimes = [];
errors = [];
% get model
n_var = model.size;
Q = model.Q;
c = model.c;
Aeq = model.Aeq;
beq = model.beq;
Aineq = model.Aineq;
bineq = model.bineq;
lb = model.lb;
ub = model.ub;
x0 = model.x0;
f_gurobi = inf;
f_quadprog = inf;
f_osqp = inf;
f_GDAMnag = inf;

%% compute a strict feasible initialization 
disp('==========================');
disp('Finding a feasible initialization');
tol = 1e-3;
lb_ini = lb + tol;
ub_ini = ub - tol;
lb_ini_0 = lb_ini;
ub_ini_0 = ub_ini;
x_init = [];
limit = 1e1;
% check strict feasibility
diff = ub-lb;
if min(abs(diff(:))) < 1e-9
    error('Modeling error in lb <= x <= ub: lb(i) == ub(i) for some i, remodel the ith bound as equalities!')
end
tic
while isempty(x_init)
    lb_ini(isinf(lb_ini_0)) = -limit;
    ub_ini(isinf(ub_ini_0)) = limit;
    options_init = optimoptions('linprog','Algorithm','interior-point');
    [x_init] = linprog([],Aineq,bineq,Aeq,beq,lb_ini,ub_ini,x0,options_init);
    if isempty(x_init)
        fprintf('Cannot find a strict feasible initialization! Trying with a higher limit value: %3.0e\n', limit);
    end
    limit = limit*10;
end
t_init = toc;
msg = ['Runtime (s): ', num2str(t_init)];
disp(msg);
%% Matlab quadprog interior-point
if any(strcmp(solvers,'quadprog'))
disp('==========================');
disp('MATLAB quadprog(interior-point-convex)');
options_quadprog = optimoptions('quadprog','Algorithm','interior-point-convex');
tic
[x_quadprog, f_quadprog, ~, output_quadprog] = quadprog(Q,c,Aineq,bineq,Aeq,beq,lb,ub,x0,options_quadprog);
iter_quadprog = output_quadprog.iterations;
t_quadprog = toc;
msg = ['Runtime (s): ', num2str(t_quadprog)];
disp(msg);
methods = [methods, "MATLAB quadprog(IPM)"];
fvals = [fvals, f_quadprog];
iters = [iters, iter_quadprog];
runtimes = [runtimes, t_quadprog];
end
%% call gurobi
if any(strcmp(solvers,'gurobi'))
disp('==========================');
disp('GUROBI');
gurobi_params = set_gurobi_parameters;
g_out = call_gurobi(model, gurobi_params);
t_gurobi = g_out.runtime;
f_gurobi = g_out.objval;
msg = ['Runtime (s): ', num2str(t_gurobi)];
disp(msg);
methods = [methods, "GUROBI"];
fvals = [fvals, f_gurobi];
iters = [iters, g_out.baritercount];
runtimes = [runtimes, t_gurobi];
end
%% osqp
if any(strcmp(solvers,'osqp'))
disp('==========================');
disp('OSQP')
% construct osqp model
m_constraints = size(bineq,1);
[A_osqp,l_osqp,u_osqp] = convert_osqp_model(Aineq,bineq,Aeq,beq,lb,ub,n_var,m_constraints);
% Create an OSQP object
prob = osqp;
% Setup workspace
prob.setup(Q, c, A_osqp, l_osqp, u_osqp);
% Solve problem
res = prob.solve();
t_osqp = res.info.run_time;
msg = ['Runtime (s): ', num2str(t_osqp)];
disp(msg);
iter_osqp = res.info.iter;
f_osqp = res.info.obj_val;
methods = [methods, "OSQP"];
fvals = [fvals, f_osqp];
iters = [iters, iter_osqp];
runtimes = [runtimes, t_osqp];
end

%% GDAM + NAG
if any(strcmp(solvers,'gdam'))
disp('==========================');
disp('GDAM + NAG (0.99)')
% check if there is an inequality constraint
if isempty(Aineq) && min(isinf(lb)) && min(isinf(ub))
    warning('GDAM: The optimization problem has no inequality constraint!')
end
% GDAM parameters
zeta = 0.99;
alpha_min = 1e-6;
% solve
tic
[x_GDAMnag,f_GDAMnag,iter_GDAMnag] = gdam_nag_sparse_QP(Q,c,Aineq,bineq,Aeq,beq,lb,ub,x_init,alpha_min,zeta, 0.99);
t_GDAMnag = toc;
msg = ['Runtime (s): ', num2str(t_GDAMnag)];
disp(msg);
methods = [methods, "GDAM + NAG (0.99)"];
fvals = [fvals, f_GDAMnag];
iters = [iters, iter_GDAMnag];
runtimes = [runtimes, t_GDAMnag];
disp('==========================');
end

%% errors
% osqp can compute infeasible solutions.
fs = [f_quadprog,f_gurobi, f_GDAMnag];
f_ref = min(fs);
error_quadprog = abs((f_quadprog - f_ref)/f_ref);
error_gurobi = abs((f_gurobi - f_ref)/f_ref);
error_gdam = abs((f_GDAMnag - f_ref)/f_ref);
error_osqp = abs((f_osqp - f_ref)/f_ref);

if any(strcmp(solvers,'quadprog'))
    errors = [errors, error_quadprog];
end
if any(strcmp(solvers,'gurobi'))
    errors = [errors, error_gurobi];
end
if any(strcmp(solvers,'osqp'))
    errors = [errors, error_osqp];
end
if any(strcmp(solvers,'gdam'))
    errors = [errors, error_gdam];
end
%%
[m,i] = min(runtimes);
scaled_runtimes= runtimes / runtimes(i);
disp(' ')
ResTable = table( fvals', iters', runtimes', errors'*100, scaled_runtimes',...
    'VariableNames',{'f(solution)','# Iterations', 'Runtime (s)','Error* (%)','Scaled runtime'}, ...
    'RowNames',methods);
disp(ResTable);
disp('*Note: a feasible solution with the lowest objective value is chosen as the reference f*.');
