function [A,l,u] = convert_osqp_model(Aineq,bineq,Aeq,beq,lb,ub,n_var, m_constraints)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


A = [Aineq;
    Aeq;
    speye(n_var)];


u = [bineq;
    beq;
    ub];

l = [-inf*ones(m_constraints, 1);
    beq;
    lb];

end

