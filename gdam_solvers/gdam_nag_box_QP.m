% Gradient Descent Akin Method with Nesterov's Accelerated Gradient
% 
% Copyright (C) 2021 
%
%     Long Chen <long.chen@tum.de>
%
function [x, f, loop] = gdam_nag_box_QP(Q,c,x0,lb,ub, alpha_min,zeta, mu)
% Initialization
gamma = 5;    % Initial step size
gamma_min = alpha_min; % minimum step size
xt = x0;
yt = xt;
loop = 0;
stopping = 1;
adapt = 0.5;
while stopping >0
    loop = loop + 1;
    %grad_f =  Q*xt + c;
    grad_f =  Q'*xt + c;  % more efficient for large sparse problems
    grad_phi = -1./(xt-lb) + 1./(ub-xt);
    s_zeta = - grad_f/norm(grad_f) - zeta*grad_phi/norm(grad_phi);
    s = s_zeta/norm(s_zeta);
    yn = xt + gamma * s;
    xn = yn + mu*(yn - yt);
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
    xt = xn;
    yt = yn; 
    if loop > 20000
        break;
    end
    if mod(loop,10) == 1
        f= 0.5* xn'*Q*xn + c'*xn;
        %fprintf('Iteration %d: f = %e \n',loop,f);
    end
end
x = yt;
f = 0.5* x'*Q*x + c'*x;
end

