% Gradient Descent Akin Method with Nesterov's Accelerated Gradient
% 
% Copyright (C) 2021 
%
%     Long Chen <long.chen@tum.de>
%
% Reference: 
% [1] L. Chen, Gradient descent akin method, PhD thesis, TU Munich, 2021
% [2] L. Chen, W. Chen, and K.-U. Bletzinger, A gradient descent 
% akin method for inequality constrained optimization, arXiv:1902.04040v4
%
% DEMO: randomly generated problems of:
%   1. box-constrained linear least squares
%   2. convex quadratic programming with equality, inequality, bounds
%   3. convex quadratic programming with equality, bounds
%   4. nonconvex box-constrained quadratic programming
%
% Support multiple runs of experiments, which result in a comparison of
% the geometric means of different solver runtime. 

function ResTable = gdam_tests(problem_type, n_var, number_runs)

addpath("problems");
addpath("gdam_solvers");
addpath("utils");

if problem_type == 1
    % create model
    M = floor(1.5*n_var);  % preset number. M >= N
    N = n_var;
    model.lb = 1.*ones(N,1);
    model.ub = 2.*ones(N,1);
    model.x0 = (model.lb + model.ub)/2.01;

    seed = 123;
    rng(seed);
    model.C = randn(M,N);
    rng(seed);
    model.d = randn(M,1);
    model.Aineq = [];
    model.bineq = [];
    model.Aeq = [];
    model.beq = [];

elseif problem_type == 2 % Benchmark credit: RACQP by Kresimir Mihic, Mingxi Zhu, and Yinyu ye
    % model parameters
    construct.n_var= n_var;
    construct.Q_sparsity= 0.8;
    construct.kappa = 0;
    construct.c_sparsity= 0;
    construct.Aeq_sparsity= 0.8;
    construct.Aineq_sparsity= 0.8;
    construct.Aineq_n_row= 10;
    construct.Aeq_n_row= 50;
    construct.rnd_seed= 123;
    
    problem.construct = construct;
    problem.type = 'construct';
    model = get_model_rnd_QP(problem.construct,true); 
    
elseif problem_type == 3 % Benchmark credit: RACQP by Kresimir Mihic, Mingxi Zhu, and Yinyu ye
    % model parameters;
    construct.n_var= n_var;
    construct.Q_sparsity= 0.1;
    construct.kappa = 0;
    construct.c_sparsity= 0;
    construct.Aeq_sparsity= 0.9;
    construct.Aineq_sparsity= 1;
    construct.Aineq_n_row= 0;
    construct.Aeq_n_row= floor(n_var/2);
    construct.rnd_seed= 123;

    problem.construct = construct;
    problem.type = 'construct';
    model = get_model_rnd_QP(problem.construct,true); 
elseif problem_type == 4
    % create model
    seed = 123;
    rng(seed);
    Q = randn(n_var);
    %Q = sprandn(n_var, n_var, 0.01);
    model.Q = 0.5*(Q+Q');
    rng(seed);
    model.c = randn(n_var,1);
    model.Aineq = [];
    model.bineq = [];
    model.Aeq = [];
    model.beq = [];
    model.lb = zeros(n_var,1);
    model.ub = ones(n_var,1);
    model.x0 = (model.lb + model.ub)/2.01;

else
end

runtime_history = [];

% multiple runs to compare the geometric mean of different solver runtime
for i = 1:number_runs
    msg = ['Run number ', num2str(i), ':'];
    disp(msg);
    disp('');
    if problem_type == 1
        [methods, fvals, iters, runtimes_lsq, errors] = boxLsqLin(model);
        runtime_history = [runtime_history; runtimes_lsq];     
    elseif problem_type == 2 || problem_type == 3
        [methods, fvals, iters, runtimes_convexQP, errors] = convexQP(model);
        runtime_history = [runtime_history; runtimes_convexQP];   
    elseif problem_type == 4
        with_fmincon = false;
        [methods, fvals, iters, runtimes_nonconvexQP, errors] = nonconvexQP(model, with_fmincon);
        runtime_history = [runtime_history; runtimes_nonconvexQP]; 
    else
        error("Problem type 1, 2, 3, and 4 accepted only")
    end
end

if number_runs > 1
    averagetime = geomean(runtime_history);
elseif number_runs == 1
    averagetime = runtime_history;
end

scaled_averagetime= averagetime / averagetime(size(averagetime,2));
disp('---------------------------------------------------------------------------------------')
if problem_type == 1
    msg = ['Results and comparisons for ', num2str(number_runs), ' run(s) of experiments'];
    disp(msg)
    disp('for box-constrained linear least squares problems')
    disp("   min  \| Cx -d \|^2")
    disp("   s.t. 1.0 <= x <= 2.0")
    msg = ['   with size(C) = (', num2str(M),',', num2str(N),')'];
    disp(msg)
    msg = ['        size(x) = (', num2str(N),',1)'];
    disp(msg)
elseif problem_type == 2 || problem_type == 3
    msg = ['Results and comparisons for ', num2str(number_runs), ' run(s) of experiments for the convex quadratic programming problem'];
    disp(msg)
    disp("min  x'Qx + c'x")
    disp("s.t. Aineq x <= bineq")
    disp("     Aeq x = beq")
    disp("     lb <= x <= ub")
    msg = ['variables n = ', num2str(n_var),  ...
        ', constraints [m_ineq, m_eq, m_lb, m_ub] = ',...
        '[',num2str(construct.Aineq_n_row),...
        ',', num2str(construct.Aeq_n_row), ...
        ',',num2str(n_var), ...
        ',',num2str(n_var),']'];
    disp(msg)
elseif problem_type == 4
    msg = ['Results and comparisons for ', num2str(number_runs), ' run(s) of experiments for the nonconvex quadratic programming problem'];
    disp(msg)
    disp("min  x'Qx + c'x")
    disp("     lb <= x <= ub")
    msg = ['variables n = ', num2str(n_var),  ...
        ', constraints [m_lb, m_ub] = ',...
        '[',num2str(n_var), ...
        ',',num2str(n_var),']'];
    disp(msg)
end

disp(' ')
ResTable = table( fvals', iters', averagetime', errors'*100, scaled_averagetime',...
    'VariableNames',{'f(solution)','# Iterations', 'Runtime (s)','Error (%)','Scaled runtime'}, ...
    'RowNames',methods);
disp(ResTable);
end