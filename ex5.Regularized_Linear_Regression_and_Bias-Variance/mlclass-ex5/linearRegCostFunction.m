function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate cost function
diff = X*theta - y;
% calculate penalty
% excluded the first theta value
theta1 = [0 ; theta(2:end, :)];
p = lambda*(theta1'*theta1);
J = (diff'*diff)/(2*m) + p/(2*m);

% calculate grads
grad = (X'*diff+lambda*theta1)/m;

% =========================================================================

grad = grad(:);

end
