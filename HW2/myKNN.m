function [ pred_class ] = myKNN(training_data, test_data, k)

X_train = training_data(:,1:end-1);
r_train = training_data(:,end);
X_test = test_data(:,1:end-1);
r_test = test_data(:,end);

for i = 1:size(X_test,1)
    
    for j = 1:size(X_train,1)
        dist(j) = sqrt(sum((X_train(j,:) - X_test(i,:)).^2));
    end
    
    [dist, inds] = sort(dist);
    
    pred_class(i) = mode(training_data(inds(1:k),end));
    
end

pred_class = pred_class';

end

