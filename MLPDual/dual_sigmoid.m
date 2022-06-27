function [sig] = dual_sigmoid(z)
%compute the sigmoid function
%sig = 1./(1+exp(-z));


% approximated version
sig = (z/(abs(z)+1)) + 0.5*ones(size(z));
sig = sig.*0.5;