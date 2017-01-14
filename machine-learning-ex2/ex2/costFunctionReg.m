function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Wang Qi updated @ 2017/01/14

tmp_sum = 0.0;
for i=1:m
  h_theta = sigmoid(X(i, :) * theta);
  tmp_sum += -y(i)*log(h_theta) - (1 - y(i))*log(1-h_theta);
endfor
J = tmp_sum/m + lambda/(2 * m) * sum(power(theta, 2));

n = length(theta);
for j=1:n
  tmp_sum = 0.0;
  for i=1:m
    h_theta = sigmoid(X(i, :) * theta);
    tmp_sum += (h_theta - y(i)) * X(i, j);
  endfor
  grad(j) = tmp_sum / m;
  
  if j!=1
    grad(j) += lambda/m*theta(j);
  endif
endfor

% Wang Qi update end


% =============================================================

end
