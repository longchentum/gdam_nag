% Gradient Descent Akin Method with Nesterov's Accelerated Gradient
%
% Comparative testing on a subset of Maros-Meszaros problems and RACQP test benchmarks. 
% Convex QPs with sparse problem matrices
%
% Solvers: - MATLAB quadprog, Gurobi, OSQP, GDAM + NAG
%
% Copyright (C) 2022 
%     Long Chen <long.chen@scicomp.uni-kl.de>
%
addpath("problems\");
addpath("gdam_solvers\");
addpath("utils\");
clear; %clc;

problem = load('data\data_cuter\CVXQP2_L.mat');
%problem = load('data\data_rnd\LCQP_N6000_SP0_95_M600_E0_5_Z2_C100_RND1.mat');
model = problem.model;
solvers = {'quadprog','gurobi','osqp','gdam'}; % Choose available solvers
ResTable = convexQP_sparse(model,solvers);
