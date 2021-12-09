% Gradient Descent Akin Method with Nesterov's Accelerated Gradient
% 
% Copyright (C) 2021 
%
%     Long Chen <long.chen@tum.de>

%clear, clc;

% problem_type
%   1. box-constrained linear least squares
%   2. convex quadratic programming with equality, inequality, and bound
%   constraints
%   3. convex quadratic programming with equality and bound constraints
%   4. nonconvex box-constrained quadratic programming
problem_type = 1;

number_variables = 2000;

% multiple runs to compare the geometric means of runtimes of different solvers
number_runs = 1;

ResTable = gdam_tests(problem_type, number_variables, number_runs);
