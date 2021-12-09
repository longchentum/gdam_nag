% Gradient Descent Akin Method with Nesterov's Accelerated Gradient
% 
% Copyright (C) 2021 
%
%     Long Chen <long.chen@tum.de>
%
% Function Gradient descent akin with Nesterov's Accelerated Gradient
function [x, f, loop] = gdam_nag_general_QP(Q,c, A, b, Aeq, beq, lb,ub, x0, alpha_min,zeta, mu)
%% Initialization
gamma = 1;    % Initial guess of the step size; Try a greedy but useful initial step size.
gamma_min = alpha_min; % minimum step size
xt = x0;
yt = xt;
loop = 0;
stopping = 1;
adapt = 0.5;
Atranspose = A';
numDim = size(x0,1);
I = eye(numDim);
% precompute projection matrix
%AA = Aeq*Aeq';
%invAA = inv(AA);
%Proj_A = Aeq'*inv(Aeq*Aeq');
%PA = Proj_A * Aeq;
%IPA = (I - Proj_A)*Aeq;
IPA = (eye(numDim)- Aeq'*inv(Aeq*Aeq') * Aeq);   % LC: quick implementation, can be numerically instable.

if isempty(lb)
    lb = -inf;
end
if isempty(ub)
    ub = inf;
end

while stopping >0
    loop = loop + 1;
    %grad_f =  Q*xt + c;
    grad_f = Q*xt + c;
    % box constraints
    grad_phi_box = -1./(xt-lb) + 1./(ub-xt);
    % linear inequality constraints
    res_ineqlin = Atranspose'*xt - b;  % the transpose trick to acclerated the sparse matrix-vector multiplication
    grad_phi_ineqlin = -(A'*(1./res_ineqlin));    % more efficient code
    %grad_phi_ineqlin = -(A'*(1./res_ineqlin));
    %grad_phi_ineqlin = -(Atranspose*(1./res_ineqlin));
    grad_phi = grad_phi_box + grad_phi_ineqlin;
    
    % projection to equality manifold
    %p_grad_f = grad_f - Proj_A * Aeq*grad_f;
    p_grad_f = IPA * grad_f;
    %p_grad_f = grad_f - Aeq'*(AA\(Aeq*grad_f));
    
    %p_grad_phi = grad_phi - Proj_A* Aeq* grad_phi;
    p_grad_phi = IPA *grad_phi;
    %p_grad_phi = grad_phi - Aeq'*(AA\(Aeq*grad_phi));

    %s_zeta = - grad_f/norm(grad_f) - zeta*grad_phi/norm(grad_phi);
    s_zeta = - p_grad_f/norm(p_grad_f) - zeta*p_grad_phi/norm(p_grad_phi);

    s = s_zeta/norm(s_zeta);
    yn = xt + gamma * s;
    xn = yn + mu*(yn - yt);   
    
    % check equality constraint violation
    %res_eq = Aeq*xn - beq;
%     if mod(loop,100) == 1 
%         f= 0.5* xn'*Q*xn + c'*xn;
%         fprintf('Iteration %d: f = %e \n',loop,f);
%     end

    % check box constraint violation
    phi_lb = -xn + lb;
    phi_ub = -ub + xn;
    if max(phi_lb) > 0 || max(phi_ub)>0
        if gamma > gamma_min
            gamma = gamma * adapt;
            yt =xt;
            continue;
        else
            break;
        end
    end

    % check linear inequality constraint violation
    res_ineqlin_next = Atranspose'*xn - b;
    if max(res_ineqlin_next) > 0
        if gamma > gamma_min
            gamma = gamma * adapt;
            yt =xt;
            continue;
        else
            break;
        end
    end

    xt = xn;
    yt = yn; 
    if loop > 10000
        break;
    end

end
x = yt;
f = 0.5* x'*Q*x + c'*x;

end