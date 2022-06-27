function [model] = mlp_fit(X, y, model)
%learn weights from training data
[w1, w2, w3] = initialize_weights(model);

l1 = model.l1;
l2 = model.l2;
eta = model.eta;
alpha = model.alpha;
epochs = model.epochs;
n_output = model.n_output;
minibatches = model.minibatches;  
decrease_const = model.decrease_const;

y_enc = encode_labels(y,n_output);
delta_w1_prev = zeros(size(w1));
delta_w2_prev = zeros(size(w2));

model.cost = Inf;

for i=1:epochs
    fprintf('Epoch: %d\n',i);
    
    %adaptive learning rate
    eta = eta / (1+decrease_const*i);
        
    %minibatch
    kk = floor(length(y)/minibatches);
    y = y(1:kk*minibatches);
    X = X(1:kk*minibatches,:);    
    I = reshape(1:length(y),minibatches,[]);
                  
    for i=1:minibatches
        display(i);
        idx = I(i,:);
        
%         %feedforward
%         [a1,z2,a2,~,a3] = feedforward(X(idx,:), w1, w2);
% 
%         %compute gradient via backpropagation
%         %[grad1, grad2] = get_gradient(a1,a2,a3,z2,y_enc(:,idx),w1,w2,l1,l2);
%         [delta_w1, delta_w2] = get_delta(a1,a2,a3,z2,y_enc(:,idx),w1,w2,l1,l2,eta);
%         
          
%         [a1,z2,a2,~,a3] = feedforward(X(idx,:), w1, w2);

        %compute gradient via backpropagation
        %[grad1, grad2] = get_gradient(a1,a2,a3,z2,y_enc(:,idx),w1,w2,l1,l2);
        [delta_w1, delta_w2, delta_w3, a4] = feedforward_and_get_delta(X(idx,:),y_enc(:,idx),w1,w2,w3,l1,l2,eta);
        
        %delta_w1 = eta * grad1;
        %delta_w2 = eta * grad2;

        w1 = w1 - delta_w1; % + (alpha * delta_w1_prev));
        w2 = w2 - delta_w2; % + (alpha * delta_w2_prev));        
        w3 = w3 - delta_w3; % + (alpha * delta_w2_prev)); 
%         delta_w1_prev = delta_w1;
%         delta_w2_prev = delta_w2;

        %cost = get_cost(y_enc(:,idx), a3, model, w1, w2);      
        %model.cost = [model.cost, cost];
        
        
    end
    
end

model.w1 = w1;
model.w2 = w2;
model.w3 = w3;

end

function [delta_w1, delta_w2, delta_w3, a4] = feedforward_and_get_delta(X,y_enc,w1,w2,w3,l1,l2,eta)
%compute gradient step using backpropagation

%compute feedforward step (to obtain  a1,a2,a3,z2 )

a1 = add_bias_unit(X, 'col');
z2 = w1*a1';
z2 = DualArray(z2,ones(size(z2)));
a2 = sigmoid(z2);
grad_a2 = getDual(a2);
a2 = getReal(a2);

a2 = add_bias_unit(a2, 'row');
z3 = w2*a2;

z3 = DualArray(z3,ones(size(z3)));
a3 = sigmoid(z3);
grad_a3 = getDual(a3);
a3 = getReal(a3);

% output neuron
a3 = add_bias_unit(a3, 'row');
z4 = w3*a3;
z4 = DualArray(z4,ones(size(z4)));
a4 = sigmoid(z4);
a4 = getReal(a4);
% compute the delta

sigma4 = a4 - y_enc;

sigma3 = (w3'*sigma4).*[1; grad_a3];
sigma3 = sigma3(2:end,:);

sigma2 = (w2'*sigma3).*[1; grad_a2];
sigma2 = sigma2(2:end,:);

grad1 = sigma2*a1;  % delta2*a1 = delta2*f(z1)=delta2*f(z1) = delta2*(a1*w0)
grad2 = sigma3*a2'; % delta3*a2 = delta3*f(z2)=delta3*f(a1*w1)
grad3 = sigma4*a3'; % delta3*a2 = delta3*f(z2)=delta3*f(a1*w1)

%regularize
%grad1(:,2:end) = grad1(:,2:end) + (w1(:,2:end)*(l1+l2));
%grad2(:,2:end) = grad2(:,2:end) + (w2(:,2:end)*(l1+l2));
%grad3(:,2:end) = grad3(:,2:end) + (w3(:,2:end)*(l1+l2));

delta_w1 = eta*grad1;
delta_w2 = eta*grad2;
delta_w3 = eta*grad3;

end


function [w1, w2, w3] = initialize_weights(model)
%initialize weights to Unif[-1,1]

n_hidden1 = model.n_hidden1;
n_hidden2 = model.n_hidden2;
n_features = model.n_features;
n_output = model.n_output;

w1 = 2*rand(1,n_hidden1*(n_features+1))-1;
w1 = reshape(w1,[n_hidden1, n_features+1]);

w2 = 2*rand(1,n_hidden2*(n_hidden1+1))-1;
w2 = reshape(w2,[n_hidden2, n_hidden1+1]);

w3 = 2*rand(1,n_output*(n_hidden2+1))-1;
w3 = reshape(w3,[n_output, n_hidden2+1]);

end

function [onehot] = encode_labels(y, k)
%one-hot encoding
onehot = zeros(k,length(y));

for i = 1:length(y)
    val = y(i)+1;   %0:9 -> 1:10
    onehot(val,i) = 1;
end

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


%% Versione originale della feedforward
function [a1,z2,a2,z3,a3] = feedforward(X, w1, w2)
%compute feedforward step

a1 = add_bias_unit(X, 'col');
z2 = w1*a1';
a2 = sigmoid(z2);
a2 = add_bias_unit(a2, 'row');
z3 = w2*a2;
a3 = sigmoid(z3);

end


%% Versione duale della feedforward (restituisce anche i gradienti)
% function [a1,z2,a2,z3,a3, grad_a2,grad_a3] = feedforward_with_gradient(X, w1, w2)
% %compute feedforward step
% 
% a1 = add_bias_unit(X, 'col');
% z2 = w1*a1';
% z2 = DualArray(z2,ones(size(z2)));
% a2 = sigmoid(z2);
% grad_a2 = a2.getDual();
% a2 = a2.getReal();
% 
% 
% a2 = add_bias_unit(a2, 'row');
% z3 = w2*a2.getReal();
% 
% z3 = DualArray(z3,ones(size(z3)));
% a3 = sigmoid(z3);
% grad_a3 = a3.getDual();
% a3 = a3.getReal();
% 
% end
% 
% function [grad1, grad2] = get_gradient(a1,a2,a3,z2,y_enc,w1,w2,l1,l2)
% %compute gradient step using backpropagation
% 
% sigma3 = a3 - y_enc;
% z2 = add_bias_unit(z2, 'row');
% sigma2 = w2'*sigma3.*sigmoid_gradient(z2);
% sigma2 = sigma2(2:end,:);
% grad1 = sigma2*a1;
% grad2 = sigma3*a2';
% 
% %regularize
% grad1(:,2:end) = grad1(:,2:end) + (w1(:,2:end)*(l1+l2));
% grad2(:,2:end) = grad2(:,2:end) + (w2(:,2:end)*(l1+l2));
% 
% end

%% VErsione originale della get_gradient
% function [grad1, grad2] = get_gradient(a1,a2,a3,z2,y_enc,w1,w2,l1,l2)
% %compute gradient step using backpropagation
% 
% sigma3 = a3 - y_enc;
% z2 = add_bias_unit(z2, 'row');
% sigma2 = w2'*sigma3.*sigmoid_gradient(z2);
% sigma2 = sigma2(2:end,:);
% grad1 = sigma2*a1;
% grad2 = sigma3*a2';
% 
% %regularize
% grad1(:,2:end) = grad1(:,2:end) + (w1(:,2:end)*(l1+l2));
% grad2(:,2:end) = grad2(:,2:end) + (w2(:,2:end)*(l1+l2));
% 
% end

function [a1,z2,a2,z3,a3, grad_a2, grad_a3] = feedforward_with_gradient(X, w1, w2)
%compute feedforward step

a1 = add_bias_unit(X, 'col');
z2 = w1*a1';
z2 = DualArray(z2,ones(size(z2)));
a2 = sigmoid(z2);
grad_a2 = a2.getDual();
a2 = a2.getReal();


a2 = add_bias_unit(a2, 'row');
z3 = w2*a2.getReal();

z3 = DualArray(z3,ones(size(z3)));
a3 = sigmoid(z3);
grad_a3 = a3.getDual();
a3 = a3.getReal();



end



function [cost] = get_cost(y_enc, output, model, w1, w2)
%compute cost function
l1 = model.l1;
l2 = model.l2;

term1 = -y_enc.*log(output);
term2 = (1-y_enc).*log(1-output);
cost = sum(term1-term2);
cost = sum(cost);
L1_term = L1_reg(l1,w1,w2);
L2_term = L2_reg(l2,w1,w2);

cost = cost + L1_term + L2_term;

end

function [cost] = L2_reg(lambda, w1, w2)
%compute l2 regularization cost
cost = (lambda/2) * (sum(sum(w1(:,2:end).^2)) + sum(sum(w2(:,2:end).^2)));
end

function [cost] = L1_reg(lambda, w1, w2)
%compute l2 regularization cost
cost = (lambda/2) * (sum(sum(abs(w1(:,2:end)))) + sum(sum(abs(w2(:,2:end)))));
end

function [sig] = sigmoid(z)
%compute the sigmoid function
%sig = 1./(1+exp(-z));


% approximated version
% sig = 0.5*(z./(1+abs(z)))+0.5*ones(size(z));
sig = (z./(abs(z)+1)) + 0.5*ones(size(z));
sig = sig.*0.5;

%sig2 = zeros(size(z));
%for i=1:length(z)
%    sig2(i) = 1/(1+exp(-z(i)));


%end

end

function [sg] = sigmoid_gradient(z)
%compute sigmoid gradient
%sig = sigmoid(z);
%sg = sig.*(1-sig);

% compute gradient of the approximated version
sg = 0.5*(1./(1+abs(z)).^2);
end