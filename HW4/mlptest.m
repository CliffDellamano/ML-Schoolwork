function [ z, test_error ] = mlptest(test_data, w, v)

% Load Data
test_data = load(test_data);

X_test = test_data(:,1:end-1);
r_test = test_data(:,end);

% Format X
X_test = [ones(size(X_test,1),1) X_test];

% Format r
r_test = [r_test zeros(size(r_test,1),size(v,1))];
for t=1:size(r_test,1)
    r_test(t,r_test(t,1)+2) = 1;
end
r_test(:,1) = [];
    
for t=1:size(X_test,1)
    % Compute forward calculation
    for h=1:size(w,1)
        z(t,h) = max(0,w(h,:)*X_test(t,:)');
    end
    for i=1:size(v,1)
        o(t,i) = sum(v(i,2:end) .* z(t,:)) + v(i,1);
    end
    for i=1:size(v,1)
        y(t,i) = (exp(o(t,i)))./(sum(exp(o(t,:))));
    end       
end
    
% Calculate test error
[~,C] = max(y,[],2);

num_errors = 0;
for t=1:size(X_test,1)
    if r_test(t,C(t)) ~= 1
        num_errors = num_errors + 1;
    end
end
error_rate = num_errors./size(X_test,1);

% Output final test error
test_error = error_rate .* 100;
sprintf('Error rate on test set : %.2f', test_error)


end



