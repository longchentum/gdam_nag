function [f,grad_f] = qp_objective(Q,c,x)

f = 0.5*x'*Q*x + c'*x;

if nargout > 1
    grad_f = Q*x + c;
end

end

