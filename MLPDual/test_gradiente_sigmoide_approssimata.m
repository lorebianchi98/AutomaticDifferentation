x = linspace(-10,10,1000);
x = x';
y = sigmoid_gradient(x);

grad_y = sigmoid_gradient_approx(x);

plot(x,y,'-b',x,grad_y,'--r');

figure;

x_dual = DualArray(x,ones(1000,1));

y_dual = sigmoid_approx(x_dual);

grad_y_dual = getDual(y_dual);

plot(x,grad_y,'--p',x,grad_y_dual,'--r');



function [sig] = sigmoid_approx(z)
% approximated version
% sig = 0.5*(z./(1+abs(z)))+0.5*ones(size(z));
sig = (z./(abs(z)+1)) + 0.5*ones(size(z));
sig = sig.*0.5;


end

function [sig] = sigmoid(z)
%compute the sigmoid function
sig = 1./(1+exp(-z));


end




function [sg] = sigmoid_gradient_approx(z)
%compute sigmoid gradient
%sig = sigmoid(z);
%sg = sig.*(1-sig);

% compute gradient of the approximated version
sg = 0.5*(1./(1+abs(z)).^2);
end


function [sg] = sigmoid_gradient(z)
%compute sigmoid gradient
sig = sigmoid(z);
sg = sig.*(1-sig);

end