%% Blank Workspace

clear
close
clc

%% Report and Plot Error Rates

z_num = [3 6 9 12 15 18];

figure
for i = 1:length(z_num)
    [ ~, w, v, train_error, val_error ] = mlptrain('optdigits_train.txt','optdigits_valid.txt',z_num(i),10);
    scatter(z_num(i), train_error, 'r')
    hold on
    scatter(z_num(i), val_error, 'b')
    drawnow
end

title('Training/Validation Error Rates')

%% Report Test Set Error Rate

% 18 hidden units usually best
[ z, test_error ] = mlptest('optdigits_test.txt', w, v);