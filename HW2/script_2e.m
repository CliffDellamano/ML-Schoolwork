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

%% Implement LDA

[ W, lambda ] = myLDA(optdigits_train, 2);

%% Project to R2

X = [X_train; X_test];
r = [r_train; r_test];

P = X * W;

%% Plot and Label Data

figure
for n = 0:9
    scatter(P(r==n,1),P(r==n,2))
    hold on
end

title('2D LDA of Optdigits Data')

labels = cellstr(num2str(r(1:15:end)));
text(P(1:15:end,1),P(1:15:end,2),labels);