%% Blank Workspace

clear
close
clc

%% Load Data

load('optdigits_train.txt');
load('optdigits_test.txt');

X_train = optdigits_train(:,1:end-1);
r_train = optdigits_train(:,end);
X_test = optdigits_test(:,1:end-1);
r_test = optdigits_test(:,end);

%% Implement PCA

[ W, lambda ] = myPCA(optdigits_train, size(X_train, 2));

%% Generate Proportion of Variance

for i=1:length(lambda)
    PoV(i) = (sum(lambda(1:i)))./(sum(lambda));
end

plot(PoV,'+r')
hold on
refline(0,0.9)
title('Proportion of Variance')
xlabel('# of Eigenvectors')
ylabel('PoV')

%% Project to K Principal Components

W = W(:,1:20);

P_train = X_train * W;
P_test = X_test * W;

%% Run KNN on Projected Data

P_train = [P_train r_train];
P_test = [P_test r_test];

k = [1,3,5,7];

for t = 1:length(k)
    pred_class = myKNN(P_train, P_test, k(t));
    
    num_errors = sum(pred_class ~= r_test);
    error_rate = (num_errors./length(pred_class)).* 100;

    sprintf('Error rate with k = %d : %.2f', k(t), error_rate)
end

