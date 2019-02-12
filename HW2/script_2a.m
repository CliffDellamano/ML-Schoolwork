%% Blank Workspace

clear
close
clc

%% Load Test Data

load('optdigits_train.txt')
load('optdigits_test.txt')

%% Implement KNN

k = [1,3,5,7];

for t = 1:length(k)
    pred_class = myKNN(optdigits_train, optdigits_test, k(t));
    
    num_errors = sum(pred_class ~= optdigits_test(:,end));
    error_rate = (num_errors./length(pred_class)).* 100;

    sprintf('Error rate with k = %d : %.2f', k(t), error_rate)
end
    
    