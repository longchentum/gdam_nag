% Gradient Descent Akin Method with Nesterov's Accelerated Gradient
% 
% Copyright (C) 2022 
%
%     Long Chen <long.chen@scicomp.uni-kl.de>
%
% Function Gradient descent akin with Nesterov's Accelerated Gradient
function [x, f, loop] = gdam_nag_sparse_QP(Q,c, A, b, Aeq, beq, lb,ub, x0, alpha_min,zeta, mu)
% Initialization
gamma_0 = pi*1e2;
gamma = gamma_0;
gamma_min = alpha_min;
max_iter = 20000;
conv_count = 3;
% Presolve
xt = x0;
yt = xt;
loop = 0;
adapt = 0.5;
Atranspose = A';
Qtranspose = Q';
AeqT = Aeq';
AA = Aeq*Aeq';
try
    dAA = decomposition(AA,"chol",'CheckCondition',false);
catch
    warning('Cholesky decomposition failed. Aeq*Aeq^T is degenerate!');
    dAA = decomposition(AA,"auto",'CheckCondition',false);
end
if isempty(lb)
    lb = -inf;
end
if isempty(ub)
    ub = inf;
end
f_last = inf;
yt_last = [];
flag_recompute = true;
msg =' iter     objective    zeta    stepsize';
disp(msg);
while loop <= max_iter
    if flag_recompute
        Qx = Qtranspose'*xt;
        grad_f = Qx + c;
        grad_phi_box = -1./(xt-lb) + 1./(ub-xt);
        res_ineqlin = Atranspose'*xt - b;  
        grad_phi_ineqlin = -(A'*(1./res_ineqlin));   
        grad_phi = grad_phi_box + grad_phi_ineqlin;
         if norm(grad_f) < 1e-9
            error('GDAM: objective gradient vanishes!');
        end
        p_grad_f = grad_f - AeqT* (dAA\(AeqT' *grad_f)); 
        if norm(grad_phi) < 1e-9
            %warning('GDAM: zero (barrier) constraint gradient!');
            s_zeta = - p_grad_f/norm(p_grad_f);
        else
            p_grad_phi = grad_phi - AeqT* (dAA\(AeqT' *grad_phi));        
            s_zeta = - p_grad_f/norm(p_grad_f) - zeta*p_grad_phi/norm(p_grad_phi);
        end
        s = s_zeta/norm(s_zeta);
    end
    yn = xt + gamma * s;
    xn = yn + mu*(yn - yt);   
    phi_lb = -xn + lb;
    phi_ub = -ub + xn;
    if max(phi_lb) > 0 || max(phi_ub)>0
        if gamma > gamma_min
            gamma = gamma * adapt;
            yt =xt;
            flag_recompute = false;
            continue;
        else
            break;
        end
    end
    res_ineqlin_next = Atranspose'*xn - b;
    if max(res_ineqlin_next) > 0
        if gamma > gamma_min
            gamma = gamma * adapt;
            yt =xt;
            flag_recompute = false;
            continue;
        else
            break;
        end
    end
    check_restart = -s'*(xn-xt);
    if check_restart > 0
        yt =xt;
        flag_recompute = false;
        continue;
    end
    if mod(loop,50) == 0
        f = xt'*(0.5*Qx + c);
        if mod(loop,100) == 0
            fprintf('%5d    %7.4e    %4.2f    %8.4f\n', loop, f, zeta, gamma);
        end
        if abs((f-f_last)/f) < 1e-5 || (f > f_last)
            conv_count = conv_count - 1;
            if conv_count <= 0
                if f> f_last
                    yt = yt_last;
                end
                break;
            end
            if (f>f_last)
                gamma= gamma* adapt^2;
            end
        end
        if abs((f-f_last)/f) < 1e-1
            gamma = gamma_0*adapt;
            yt =xt;
            flag_recompute = false;
            loop = loop +1;
            %fprintf('adapt gamma!\n');
            f_last = f;
            yt_last = yt;
            continue;
        end
        f_last = f;
        yt_last = yt;
    end
    xt = xn;
    yt = yn; 
    loop = loop + 1;
    flag_recompute = true;
end
x = yt;
f = 0.5* x'*(Qtranspose'*x) + c'*x;
end