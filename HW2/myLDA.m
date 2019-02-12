function [ W, lambda ] = myLDA(data, num_principal_components)

X = data(:,1:end-1);
r = data(:,end);

% Calculate between-class scatter

for i = 1:10
    mu(i,:) = (sum(X(r==(i-1),:)))./(sum(r==(i-1)));
end

mu_global = sum(mu)./10;

Sb = zeros(size(X,2));

for i=1:10
    Sb = Sb + (mu(i,:)-mu_global)'*(mu(i,:)-mu_global)*(sum(r==(i-1)));
end

% Calculate within-class scatter

Sw = zeros(size(X,2));

for i=1:10 
    
    Si = zeros(size(X,2));
    
    for t=1:size(X,1)
        Si = Si + (X(t,:)-mu(i,:))'*(X(t,:)-mu(i,:))*(r(t)==(i-1));
    end
    
    Sw = Sw + Si;
    
end

% Find best W

[W, D] = eig(pinv(Sw)*Sb);
[D, inds] = sort(diag(D), 'descend');
W = W(:,inds);

% Return values

W = W(:,1:num_principal_components);
lambda = D(1:num_principal_components);

end

