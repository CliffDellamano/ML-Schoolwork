function [ W, lambda ] = myPCA(data, num_principal_components)

X = data(:,1:end-1);

% Center data

mu = mean(X);
X = X - mu;

% Calculate covariance matrix

S = (X'*X)./size(X,1);

% Perform eigen-decomposition

[W, D] = eig(S);
[D, inds] = sort(diag(D), 'descend');
W = W(:,inds);

% Return values

W = W(:,1:num_principal_components);
lambda = D(1:num_principal_components);

end

