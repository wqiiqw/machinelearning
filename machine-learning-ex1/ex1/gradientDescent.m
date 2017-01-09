function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% Debugging shows that if mydiff is > 0.01, the submission will fail. So set to mydiff to be 0.01 or smaller
mydiff = 0.01;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % xfunction = theta(1) + theta(2) * X(iter, 2)
    % diff = xfunction - y(iter, 1)
	
	% Gradient descent on theta1
	temp1 = 0.0;
    for iter2 = 1:m
	    temp1 = temp1 + (theta(1) + theta(2) * X(iter2, 2) - y(iter2, 1));
	end
	temp1 =  temp1 / m;
  
	% Gradient descent on theta2
	temp2 = 0.0;
    for iter2 = 1:m
	    temp2 = temp2 + (theta(1) + theta(2) * X(iter2, 2) - y(iter2, 1)) * X(iter2, 2);
	end
	temp2 =  temp2 / m;

	% Do the simutaneous updates for both theta1 and theta2
	if abs(temp1) > mydiff
		% didn't converge, then update
		theta(1) = theta(1) - alpha * temp1 ;
	end
	
	if abs(temp2) > mydiff
		% didn't converge, then update
		theta(2) = theta(2) - alpha * temp2;
	end


  %fprintf('theta2 is %d and temp is %d ...\n', theta(2), temp)


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	
end
    %fprintf('Scott Program paused. Press enter to continue.\n');
	%pause
    %J_history
end