function [J, grad] = alrCostFunction(theta, X, y, lambda)
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
%
% Compute the Cost
tmp_sum = 0.0;
for i=1:m
  h_theta = sigmoid(X(i, :) * theta);
  tmp_sum += -y(i)*log(h_theta) - (1 - y(i))*log(1-h_theta);
endfor

% below "2:end" *In the Ex2.pdf note, Page 9, the doc mentioned that: "Note that 
% you should not regularize the parameter theta0. In Octave/MATLAB, recall that 
% the index starts from 1, hence, you should not be regularizing the theta1 
% parameter (which corresponds to theta0) in the code. so the change is done.
J = tmp_sum/m + lambda/(2 * m) * sum(power(theta(2:end), 2));

% Compute the gradient
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

