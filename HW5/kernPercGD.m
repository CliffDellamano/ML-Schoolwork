function [alpha, b] = kernPercGD(train_data, train_label)

% Initialize parameters
N = size(train_data,1);

alpha = zeros(N,1);
b = 0;

% Set convergence
errors = 0;
prev_error = 0;
current_error = 1;

while abs(prev_error - current_error) > 0.01
    
    for t=1:N
        
        pred_sum = 0;
        for i=1:N
            pred = alpha(i) .* train_label(i) .* ((train_data(i,:) * train_data(t,:)').^2);
            pred_sum = pred_sum + pred;
        end
        
        if (pred_sum + b) * train_label(t) <= 0
%             errors = errors + 1;          
            alpha(t) = alpha(t) + 1;
            b = b + train_label(t);
        end
        
    end
    
    for t=1:N
        
        pred_sum = 0;
        for i=1:N
            pred = alpha(i) .* train_label(i) .* ((train_data(i,:) * train_data(t,:)').^2);
            pred_sum = pred_sum + pred;
        end
        
        if (pred_sum + b) * train_label(t) <= 0
            errors = errors + 1; 
        end
    end
        
    
    prev_error = current_error;
    current_error = errors./N;
    errors = 0;

end

sprintf('Error rate on training dataset: %.2f', current_error * 100)

end

