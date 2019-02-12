%% Blank Workspace

clear
%close
clc

%% Load Data

load('face_train_data_960.txt')
load('face_test_data_960.txt')

data = [face_train_data_960; face_test_data_960];

%% Implement PCA

[ W, lambda ] = myPCA(data, size(data, 2)-1);

%% Visualize Eigenfaces

figure
for i = 1:5
    subplot(1,5,i)
    imagesc(reshape(W(:,i),32,30)')
    
    if i ==3
        title('First 5 Eigenfaces')
    end
end