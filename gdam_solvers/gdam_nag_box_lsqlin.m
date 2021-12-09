% Gradient Descent Akin Method with Nesterov's Accelerated Gradient
% 
% Copyright (C) 2021 
%
%     Long Chen <long.chen@tum.de>
%
function [x, f, loop] = gdam_nag_box_lsqlin(A,b,x0,lb,ub,alpha_min, zeta, mu)
%% Initialization
%zeta = 0.999;    % zeta
% Nesterov's Accleration
gamma = 1;    % Initial step size
gamma_min = alpha_min; % minimum step size
xt = x0;
yt = xt;
loop = 0;
stopping = 1;
adapt = 0.5;
AA = A'*A;
Ab = A'*b;
f_last =  norm(A*xt-b)^2;
while stopping >0
    loop = loop + 1;
    grad_f = 2*(AA*xt - Ab);
    grad_phi = -1./(xt-lb)+ 1./(ub-xt);
    % compute the gradient descent akin direction
    s_zeta = - grad_f/norm(grad_f) - zeta*grad_phi/norm(grad_phi);
    s = s_zeta/norm(s_zeta);    
    % update optimization variable
    yn = xt + gamma * s;
    xn = yn + mu*(yn - yt);
    
    if mod(loop,10) == 1
        f= norm(A*yn-b)^2;
%         if mod(loop,100) == 1
%             fprintf('Iteration %d: f = %e \n',loop,f);
%         end
        if f > f_last
            if gamma > gamma_min
                gamma = gamma * adapt^3;
                yt =xt;
                continue;
            else
                break;
            end
        end
    end

    f_last = f;

    % check constraint violations
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
   
    % go to the next step
    xt = xn;
    yt = yn; 

    if loop > 20000
        break;
    end
end
x = yt;
f = norm(A*yt - b)^2;
end

