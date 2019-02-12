%% Blank Workspace

clear
close
clc

%% Load Data

load('face_train_data_960.txt')

X_train = face_train_data_960(:,1:end-1);
r_train = face_train_data_960(:,end);

%% Project and Backproject

K = [10, 50, 100];
mu = mean(X_train);

figure
for p = 1:5
    subplot(4,5,p)
    imagesc(reshape(X_train(p,:),32,30)')
    hold on
end

for i = 1:length(K)
    
    [ W, lambda ] = myPCA(face_train_data_960, K(i));
    
    for j = 1:5
        P_train(j,:) = X_train(j,:) * W;
        X_hat(j,:) = P_train(j,:)*W' + mu;
        
        subplot(4,5,(5*i)+j)
        imagesc(reshape(X_hat(j,:),32,30)')
    end
    
    clearvars P_train X_hat
    
end
    
    

