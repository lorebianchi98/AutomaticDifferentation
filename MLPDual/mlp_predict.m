function [y_pred] = mlp_predict(X, model)
%predict class labels
w1 = model.w1;
w2 = model.w2;
w3 = model.w3;

[a1,z2,a2,z3,a3,z4,a4] = feedforward(X, w1, w2, w3);
[~, argmax] = max(z4);
y_pred = argmax - 1;  %1:10 -> 0:9

end

function [a1,z2,a2,z3,a3,z4,a4] = feedforward(X, w1, w2, w3)
%compute feedforward step

a1 = add_bias_unit(X, 'col');
z2 = w1*a1';
a2 = sigmoid(z2);
a2 = add_bias_unit(a2, 'row');
z3 = w2*a2;
a3 = sigmoid(z3);
a3 = add_bias_unit(a3, 'row');
z4 = w3*a3;
a4 = sigmoid(z4);

end

function [X_new] = add_bias_unit(X, how)
%add a bias unit (col or row of 1s) to array at index 1

if strcmp(how,'col')
    X_new = ones(size(X,1), size(X,2)+1);
    X_new(:,2:end) = X;
elseif strcmp(how,'row')
    X_new = ones(size(X,1)+1, size(X,2));
    X_new(2:end,:) = X;
else
    fprintf('how must be row or col!\n')
    X_new = X;
end

end

function [sig] = sigmoid(z)
%compute the sigmoid function
sig = (z./(abs(z)+1)) + 0.5*ones(size(z));
sig = sig.*0.5;
end

