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

%output of hypothesis
hypothesis = sigmoid(X * theta);
%compute the J,not regularize the parameter theta_0(in the matlab,the fist element of theta is indexed as 1)
J = sum(-y'*log(hypothesis)-(1-y)'*log(1-hypothesis)) / m + (sum(theta(2:size(theta)) .^ 2)) * lambda / (2 * m);

%compute the gradient
%formula of grad_1 is different from the other gradient.
grad(1) = sum((hypothesis - y)' *  X(:,1)) / m;
for theta_index = 2: size(X,2);
    grad(theta_index) = sum((hypothesis - y)' *  X(:,theta_index)) / m + lambda / m * theta(theta_index);
end





% =============================================================

end
