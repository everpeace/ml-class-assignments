function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
%
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% initial version.
% for i=1:m
  % dif = (theta'*X(i, :)'-y(i));
  % J = J + dif*dif;
% endfor
% J = J / (2*m);

% vectorized version.
% (exactly the same with multivariate version. )
dif = X*theta -y;
J = (dif'*dif)/(2*m);
%
% =========================================================================

end
