function [ z, w, v, train_error, val_error ] = mlptrain(train_data, val_data, m, k)

% Load Data
train_data = load(train_data);
val_data = load(val_data);

X_train = train_data(:,1:end-1);
r_train = train_data(:,end);
X_val = val_data(:,1:end-1);
r_val = val_data(:,end);

% Format X
X_train = [ones(size(X_train,1),1) X_train];
X_val = [ones(size(X_val,1),1) X_val];

% Format r
r_train = [r_train zeros(size(r_train,1),k)];
r_val = [r_val zeros(size(r_val,1),k)];
for t=1:size(r_train,1)
    r_train(t,r_train(t,1)+2) = 1;
end
for t=1:size(r_val,1)
    r_val(t,r_val(t,1)+2) = 1;
end
r_train(:,1) = [];
r_val(:,1) = [];

% Initialize v's and w's
v = -0.01 + 0.02 .* rand([k m+1]);
w = -0.01 + 0.02 .* rand([m 65]);

% Initialize learning rate
eta = 10e-4;

% Set convergence condition
iter_diff = 1;
prev_rate = 0;
num_iter = 0;

% Randomize training set
rand_order = randperm(size(X_train,1));
X_train = X_train(rand_order,:);
r_train = r_train(rand_order,:);

while iter_diff > 0.01 && num_iter < 25
    
    for t=1:size(X_train,1)
        % Compute forward calculation
        for h=1:m
            z(t,h) = max(0,w(h,:)*X_train(t,:)');
        end
        for i=1:k
            o(t,i) = sum(v(i,2:end) .* z(t,:)) + v(i,1);
        end
        for i=1:k
            y(t,i) = (exp(o(t,i)))./(sum(exp(o(t,:))));
        end

        % Compute backward calculation
        for i=1:k
            delta_v(i,1) =  eta .* (r_train(t,i) - y(t,i));
            for h=1:m
                delta_v(i,h+1) = eta .* (r_train(t,i) - y(t,i)) .* z(t,h);
            end
        end
        for h=1:m
            for j=1:size(X_train,2)
                if z(t,h) > 0
                    delta_w(h,j) = eta .* (sum((r_train(t,:) - y(t,:)) .* v(:,h)')) .* X_train(t,j);
                else
                    delta_w(h,j) = 0;
                end
            end
        end

        % Update v's and w's
        v = v + delta_v;
        w = w + delta_w;
    end
    
    % Calculate training error
    [~,C] = max(y,[],2);
    
    num_errors = 0;
    for t=1:size(X_train,1)
        if r_train(t,C(t)) ~= 1
            num_errors = num_errors + 1;
        end
    end
    error_rate = num_errors./size(X_train,1);
    iter_diff = abs(prev_rate - error_rate);
    prev_rate = error_rate;
    
    num_iter = num_iter + 1;
    
%         % Plot error rate
%     scatter(num_iter, error_rate)
%     hold on
%     drawnow

    % Clear outputs
    clear o y z delta_v delta_w 
    
end

% Output final training error
train_error = error_rate .* 100;
sprintf('Error rate on training set : %.2f', train_error)

% Clear leftovers
clear C error_rate iter_diff num_errors num_iter prev_rate r_train rand_order ...
    train_data X_train

% Randomize validation set
rand_order = randperm(size(X_val,1));
X_val = X_val(rand_order,:);
r_val = r_val(rand_order,:);

% Forward calculation - validation set
for t=1:size(X_val,1)
    for h=1:m
        z(t,h) = max(0,w(h,:)*X_val(t,:)');
    end
    for i=1:k
        o(t,i) = sum(v(i,2:end) .* z(t,:)) + v(i,1);
    end
    for i=1:k
        y(t,i) = (exp(o(t,i)))./(sum(exp(o(t,:))));
    end
end

% Calculate validation error
[~,C] = max(y,[],2);

num_errors = 0;
for t=1:size(X_val,1)
    if r_val(t,C(t)) ~= 1
        num_errors = num_errors + 1;
    end
end
error_rate = num_errors./size(X_val,1);

% Output final validation error
val_error = error_rate .* 100;
sprintf('Error rate on validation set : %.2f', val_error)

end

