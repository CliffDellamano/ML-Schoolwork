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

%% Implement LDA and KNN

L = [2, 4, 9];
k = [1, 3, 5];

for i = 1:length(L)
    
    [ W, lambda ] = myLDA(optdigits_train, L(i));
    
    P_train = X_train * W;
    P_test = X_test * W;
    
    P_train = [P_train r_train];
    P_test = [P_test r_test];
    
    for j = 1:length(k)
        
        pred_class = myKNN(P_train, P_test, k(j));
    
        num_errors = sum(pred_class ~= r_test);
        error_rate = (num_errors./length(pred_class)).* 100;

        sprintf('Error rate with k = %d and L = %d : %.2f', k(j), L(i), error_rate)
        
    end
    
    clearvars P_train P_test 
    
end