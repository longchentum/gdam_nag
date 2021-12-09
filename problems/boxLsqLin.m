% Bechmark: Gradient descent akin method for box-constrained nonconvex QP.
% This is an implementation of the gradient descent akin method (GDAM)
% with Nesterov's Accleration Gradient (NAG) for the box-constrained
% linear least squares problem
%
% min \| Cx -d \|^2, s.t. 1.0 \leq x \leq 2.0
%
% The results obtained by GDAM and GDAM+NAG are compared with the MATLAB
% solver lsqlin
%
% Author: Long Chen,
%
% Acknowledgement: Steve Schäfer, Andreas Apostolatos @MathWorks
%
% Reference: L. Chen, W. Chen, and K.-U. Bletzinger, A gradient descent 
% akin method for inequality constrained optimization, arXiv:1902.04040v4
%
%clear;

function [methods, fvals, iters, runtimes, errors] = boxLsqLin(model)

% Initialization
methods = [];
fvals = [];
iters = [];
runtimes = [];
errors = [];

%% problem definition
x0 = model.x0;
% random initilialization
C = model.C; 
d = model.d;
A = model.Aineq;
b = model.bineq;
Aeq = model.Aeq;
beq = model.beq;
lb = model.lb;
ub = model.ub;


%% MATLAB lsqlin:'interior-point'
disp('==========================');
disp('MATLAB lsqlin:interior-point');
options = optimoptions('lsqlin','Algorithm','interior-point');
tic
[x_IPM, fval_IPM, ~] = lsqlin(C,d,A,b,Aeq,beq,lb,ub, x0,options);
t_IPM = toc;
msg = ['Runtime (s): ', num2str(t_IPM)];
disp(msg);
methods = [methods, "MATLAB lsqlin(IPM)"];
fvals = [fvals, fval_IPM];
iters = [iters, NaN];
runtimes = [runtimes, t_IPM];
errors = [errors, 0];


%% MATLAB lsqlin:'trust-region-reflective'
disp('==========================');
disp('MATLAB lsqlin:trust-region-reflective');
options = optimoptions('lsqlin','Algorithm','trust-region-reflective');
tic
[x_TRR, fval_TRR, ~] = lsqlin(C,d,A,b,Aeq,beq,lb,ub,x0,options);
t_TRR = toc;
msg = ['Runtime (s): ', num2str(t_TRR)];
disp(msg);
methods = [methods, "MATLAB lsqlin(TRR)"];
fvals = [fvals, fval_TRR];
iters = [iters, NaN];
runtimes = [runtimes, t_TRR];
errors = [errors, abs((fval_TRR-fval_IPM)/fval_IPM)];

%% osqp
disp('==========================');
disp('OSQP');
% convert LSP QP
Q = C'*C;
q = -C'*d;
Aineq = [];
bineq = [];
Aeq = [];
beq = [];
n_var = size(x0,1);
m_constraints = size(bineq,1);
[A_osqp,l_osqp,u_osqp] = convert_osqp_model(Aineq,bineq,Aeq,beq,lb,ub,n_var,m_constraints);
% Create an OSQP object
prob = osqp;
% Setup workspace
prob.setup(Q, q, A_osqp, l_osqp, u_osqp);
% Solve problem
res = prob.solve();
t_osqp = res.info.run_time;
msg = ['Runtime (s): ', num2str(t_osqp)];
disp(msg);
iter_osqp = res.info.iter;
x_osqp = res.x;
f_osqp = norm(C*x_osqp - d)^2;
methods = [methods, "OSQP"];
fvals = [fvals, f_osqp];
iters = [iters, iter_osqp];
runtimes = [runtimes, t_osqp];
errors = [errors, abs((f_osqp-fval_IPM)/fval_IPM)];


%% GDAM + NAG, zeta = 0.99
disp('==========================');
disp('GDAM+NAG, zeta = 0.99');
alpha_min = 1e-3;
zeta = 0.99;
mu = 0.99;
tic
[x_GDAMnag, fval_GDAMnag, iter_GDAMnag] = gdam_nag_box_lsqlin(C,d,x0,lb, ub, alpha_min, zeta, mu);
t_GDAMnag = toc;
msg = ['Runtime (s): ', num2str(t_GDAMnag)];
disp(msg);
methods = [methods, "GDAM + NAG (0.99)"];
fvals = [fvals, fval_GDAMnag];
iters = [iters, iter_GDAMnag];
runtimes = [runtimes, t_GDAMnag];
errors = [errors, abs((fval_GDAMnag-fval_IPM)/fval_IPM)];
disp('==========================');

end