%% Bechmark: Gradient descent akin method for box-constrained nonconvex QP.
% This is an implementation of the gradient descent akin method (GDAM)
% with Nesterov's Accleration Gradient (NAG) for the box-constrained
% nonconvex Quadratic Programming problem
%
% min 1/2*x^T Q x + c^T x, s.t. 0.0 <= x <= 1.0
%
% The results obtained by GDAM and GDAM+NAG are compared with the MATLAB
% QP solver quadprog
%
% Author: Long Chen,
%
% Acknowledgement: Steve Schäfer, Andreas Apostolatos @MathWorks
%
% Reference: L. Chen, W. Chen, and K.-U. Bletzinger, A gradient descent 
% akin method for inequality constrained optimization, arXiv:1902.04040v4
%
function [methods, fvals, iters, runtimes, errors] = nonconvexQP(model, with_fmincon)
% Initialization
methods = [];
fvals = [];
iters = [];
runtimes = [];
errors = [];

% get problem
Q = model.Q;
c = model.c;
AA = model.Aineq;
bb = model.bineq;
Aeq = model.Aeq;
beq = model.beq;
lb = model.lb;
ub = model.ub;
x0 = model.x0;

%% MATLAB quadprog
disp('==========================');
disp('MATLAB quadprog');
options = optimoptions('quadprog','Display','iter','Algorithm','trust-region-reflective');
tic
[x_quadprog, f_quadprog, ~, output_quadprog]  = quadprog(Q,c,AA,bb,Aeq,beq,lb,ub,x0,options);
t_quadprog = toc;
iter_quadprog = output_quadprog.iterations;
msg = ['Runtime (s): ', num2str(t_quadprog)];
disp(msg);
error_quadprog = 0;

methods = [methods, "MATLAB quadprog(TRR)"];
fvals = [fvals, f_quadprog];
iters = [iters, iter_quadprog];
runtimes = [runtimes, t_quadprog];
errors = [errors, error_quadprog];

%% MATLAB fmincon

if with_fmincon
    disp('==========================');
    %n_var = size(x0,1);
    nonlcon = [];
    fun = @(x) qp_objective(Q,c,x);
    options_IPM = optimoptions('fmincon','Display','iter','Algorithm','interior-point','MaxIter',5000,'MaxFunEvals',50000,'GradObj','on');
    
    tic
    [x_fmincon,f_fmincon,~,output_fmincon] = fmincon(fun,x0,AA,bb,Aeq,beq,lb,ub,nonlcon, options_IPM);
    t_fmincon = toc;
    iter_fmincon = output_fmincon.iterations;
    msg = ['Runtime (s): ', num2str(t_fmincon)];
    disp(msg);
    error_fmincon = abs((f_fmincon-f_quadprog)/f_quadprog);
    
    methods = [methods, "MATLAB fmincon(IPM)"];
    fvals = [fvals, f_fmincon];
    iters = [iters, iter_fmincon];
    runtimes = [runtimes, t_fmincon];
    errors = [errors, error_fmincon];
else
    methods = [methods, "MATLAB fmincon(IPM)"];
    fvals = [fvals, NaN];
    iters = [iters, NaN];
    runtimes = [runtimes, NaN];
    errors = [errors, NaN];
end

%% GDAM + NAG, zeta = 0.999
disp('==========================');
disp('GDAM+NAG, zeta = 0.999');
alpha_min = 1e-4;
zeta = 0.999;
mu = 0.99;
tic
[x_GDAMnag_1, fval_GDAMnag_1, iter_GDAMnag_1] = gdaM_nag_box_QP(Q,c,x0,lb,ub, alpha_min,zeta, mu);
t_GDAMnag_1 = toc;
msg = ['Runtime (s): ', num2str(t_GDAMnag_1)];
disp(msg);
error_GDAMnag_1 = abs((fval_GDAMnag_1-f_quadprog)/f_quadprog);
methods = [methods, "GDAM + NAG (0.999)"];
fvals = [fvals, fval_GDAMnag_1];
iters = [iters, iter_GDAMnag_1];
runtimes = [runtimes, t_GDAMnag_1];
errors = [errors, error_GDAMnag_1];
%% GDAM + NAG, zeta = 0.99
disp('==========================');
disp('GDAM+NAG, zeta = 0.99');
zeta = 0.99;
mu = 0.99;
tic
[x_GDAMnag_2, fval_GDAMnag_2, iter_GDAMnag_2] = gdaM_nag_box_QP(Q,c,x0,lb,ub, alpha_min,zeta,mu);
t_GDAMnag_2 = toc;
msg = ['Runtime (s): ', num2str(t_GDAMnag_2)];
disp(msg);
error_GDAMnag_2 = abs((fval_GDAMnag_2-f_quadprog)/f_quadprog);
methods = [methods, "GDAM + NAG (0.99)"];
fvals = [fvals, fval_GDAMnag_2];
iters = [iters, iter_GDAMnag_2];
runtimes = [runtimes, t_GDAMnag_2];
errors = [errors, error_GDAMnag_2];




