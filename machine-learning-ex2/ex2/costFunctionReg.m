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

theta_rmv_f = theta(:,:);
theta_rmv_f(1) = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%hypothesis 
h = sigmoid (X * theta);
%Cost Function
J =( - y' * log(h)  - ((1-y)' * log(1-h)) )/ m + lambda./(2*m) * sum(theta_rmv_f .* theta_rmv_f );
%Gradient Descent
grad = X' * (h - y) / m  + (lambda/ m * theta_rmv_f);

% =============================================================

end
