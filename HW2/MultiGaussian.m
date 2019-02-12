function [ learned_params, error_rate ] = MultiGaussian(training_data, testing_data, Model)

training_data = load(training_data);
test_data = load(testing_data);

X_train = training_data(:,1:end-1);
r_train = [training_data(:,end) == 1, training_data(:,end) == 2];
X_test = test_data(:,1:end-1);
r_test = [test_data(:,end) ==1, test_data(:,end) == 2];

mu_1 = sum(X_train.*r_train(:,1))./(sum(r_train(:,1)));
mu_2 = sum(X_train.*r_train(:,2))./(sum(r_train(:,2)));

P1 = sum(r_train(:,1))./length(r_train);
P2 = sum(r_train(:,2))./length(r_train);

learned_params = {P1, P2, mu_1, mu_2};

if Model == 1
    
    S1 = zeros(size(X_train,2));
    S2 = zeros(size(X_train,2));

    for t=1:length(X_train)
        S1 = S1 + (X_train(t,:)-mu_1)'*(X_train(t,:)-mu_1)*r_train(t,1);
        S2 = S2 + (X_train(t,:)-mu_2)'*(X_train(t,:)-mu_2)*r_train(t,2);
    end

    S1 = S1./sum(r_train(:,1));
    S2 = S2./sum(r_train(:,2));
    
    learned_params{end+1} = S1;
    learned_params{end+1} = S2;

elseif Model == 2
    
    sig1 = zeros(size(X_train,2));
    sig2 = zeros(size(X_train,2));

    for t=1:length(X_train)
        sig1 = sig1 + (X_train(t,:)-mu_1)'*(X_train(t,:)-mu_1)*r_train(t,1);
        sig2 = sig2 + (X_train(t,:)-mu_2)'*(X_train(t,:)-mu_2)*r_train(t,2);
    end

    sig1 = sig1./sum(r_train(:,1));
    sig2 = sig2./sum(r_train(:,2));  
    
    S = P1 * sig1 + P2 * sig2;
    
    S1 = S;
    S2 = S;
    
    learned_params{end+1} = S1;
    learned_params{end+1} = S2;
    
elseif Model == 3
    
    alpha1 = 0;
    alpha2 = 0;
    
    for t=1:length(X_train)
        alpha1 = alpha1 + (X_train(t,:)-mu_1)*(X_train(t,:)-mu_1)'*r_train(t,1);
        alpha2 = alpha2 + (X_train(t,:)-mu_2)*(X_train(t,:)-mu_2)'*r_train(t,2);
    end
    
    alpha1 = alpha1./sum(r_train(:,1));
    alpha2 = alpha2./sum(r_train(:,2));
    
    S1 = alpha1 *eye(size(X_train,2));
    S2 = alpha2 *eye(size(X_train,2));
    
    learned_params{end+1} = alpha1;
    learned_params{end+1} = alpha2;
    
end

pred_class = zeros(size(X_test,1),2);

for n = 1:size(X_test,1)
    
    Px_1(n) = ((1)/(((2*pi)^(size(X_test,2)/2))*((det(S1))^(1/2)))) * exp((-1/2)*(X_test(n,:) - mu_1)*inv(S1)*(X_test(n,:) - mu_1)');
    Px_2(n) = ((1)/(((2*pi)^(size(X_test,2)/2))*((det(S2))^(1/2)))) * exp((-1/2)*(X_test(n,:) - mu_2)*inv(S2)*(X_test(n,:) - mu_2)');
    
%     Px_1(n) = mvnpdf(X_test(n,:), mu_1, S1);
%     Px_2(n) = mvnpdf(X_test(n,:), mu_2, S2);
    
    log_odds(n) = log((P1*Px_1(n))/(P2*Px_2(n)));
    
    if log_odds(n) > 0
        pred_class(n,1) = 1;
    elseif log_odds(n) < 0
        pred_class(n,2) = 1;
    end
end
        
num_errors = sum(sum(pred_class(:,1) ~= r_test(:,1)));
error_rate = (num_errors./length(pred_class)).* 100;

sprintf('Error rate with Model %d : %.2f', Model, error_rate)

end




