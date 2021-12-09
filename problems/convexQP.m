% Bechmark: Gradient descent akin method for linear programming problem.
% This is an implementation of the gradient descent akin method (GDAM)
% with Nesterov's Accleration Gradient (NAG) for general QP.
%
% min  1/2 x^T Q x + c^T x, 
% s.t. Aineq x <= bineq
%      Aeq x = beq
%      lb <= x <= ub
%
% The results obtained by GDAM+NAG are compared with the MATLAB quadprog
% and the opensource QP solver OSQP 
%
% Author: Long Chen,
%
% Reference: L. Chen, W. Chen, and K.-U. Bletzinger, A gradient descent 
% akin method for inequality constrained optimization, arXiv:1902.04040v4
%
%clear;

function [methods, fvals, iters, runtimes, errors] = convexQP(model)
% Initialization
methods = [];
fvals = [];
iters = [];
runtimes = [];
errors = [];

% get problem
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

%% compute a feasible initialization
disp('==========================');
disp('Finding a feasible initialization');
tic
options_init = optimoptions('linprog','Algorithm','interior-point');
[x_init] = linprog([],Aineq,bineq,Aeq,beq,lb,ub,x0,options_init);
t_init = toc;
msg = ['Runtime (s): ', num2str(t_init)];
disp(msg);

%% Matlab quadprog interior-point
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
errors = [errors, 0];

%% osqp
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
error_osqp = abs((f_osqp - f_quadprog)/f_quadprog);

methods = [methods, "OSQP"];
fvals = [fvals, f_osqp];
iters = [iters, iter_osqp];
runtimes = [runtimes, t_osqp];
errors = [errors, error_osqp];


%% GDAM + NAG
disp('==========================');
disp('GDAM + NAG (0.99)')
% GDAM parameters
zeta = 0.99;
alpha_min = 1e-4;

tic
[x_GDAMnag,f_GDAMnag,iter_GDAMnag] = gdam_nag_general_QP(Q,c,Aineq,bineq,Aeq,beq,lb,ub,x_init,alpha_min,zeta, 0.99);
t_GDAMnag = toc;
msg = ['Runtime (s): ', num2str(t_GDAMnag)];
disp(msg);
error_GDAMnag = abs((f_GDAMnag-f_quadprog)/f_quadprog);
methods = [methods, "GDAM + NAG (0.99)"];
fvals = [fvals, f_GDAMnag];
iters = [iters, iter_GDAMnag];
runtimes = [runtimes, t_GDAMnag];
errors = [errors, error_GDAMnag];
disp('==========================');

end