%% Blank Workspace

clear
close
clc

%% Train MLP

[ ~, w, v, train_error, val_error ] = mlptrain('optdigits_train.txt','optdigits_valid.txt',18,10);

train_data = load('optdigits_train.txt');
val_data = load('optdigits_valid.txt');

test_data = [train_data; val_data];
dlmwrite('optdigits_train_valid.txt', test_data);

%% Test MLP

[ z, test_error ] = mlptest('optdigits_train_valid.txt', w, v);

%% Perform 2-D PCA

X_test = test_data(:,1:end-1);
r_test = test_data(:,end);

z_prin = pca(z);
z_proj_2 = z * z_prin(:,1:2);

figure
for n = 0:9
    scatter(z_proj_2(r_test==n,1),z_proj_2(r_test==n,2))
    hold on
end

title('2D PCA of Optdigits Data')
xlabel('Hidden 1')
ylabel('Hidden 2')

labels = cellstr(num2str(r_test(1:15:end)));
text(z_proj_2(1:15:end,1),z_proj_2(1:15:end,2),labels);

%% Perform 3-D PCA

z_proj_3 = z * z_prin(:,1:3);

figure
for n = 0:9
    scatter3(z_proj_3(r_test==n,1),z_proj_3(r_test==n,2),z_proj_3(r_test==n,3))
    hold on
end

title('3D PCA of Optdigits Data')
xlabel('Hidden 1')
ylabel('Hidden 2')
zlabel('Hidden 3')

labels = cellstr(num2str(r_test(1:15:end)));
text(z_proj_3(1:15:end,1),z_proj_3(1:15:end,2),z_proj_3(1:15:end,3),labels);