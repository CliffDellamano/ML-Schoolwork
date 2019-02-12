%% Blank Workspace

clear
close
clc

%% Create Data

rng(1); % For reproducibility

r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1); % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

r2 = sqrt(3*rand(100,1)+2); % Radius
t2 = 2*pi*rand(100,1); % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

%% Visualize Data

figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

%% Combine Data and Assign Labels

data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;

%% Implement Kernel Perceptron

[alpha, b] = kernPercGD(data3, theclass);

%% Plot Decision Boundary

d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

for i = 1:size(xGrid,1)
    
    for t=1:size(alpha,1)
        w_i(t) = alpha(t) .* theclass(t) .* (data3(t,:) * xGrid(i,:)').^2;
    end    
    
    if sum(w_i) + b > 0
        xGrid_label(i) = 1;
    else
        xGrid_label(i) = -1;
    end
    
end

figure;
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
contour(x1Grid,x2Grid,reshape(xGrid_label,size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1'});
axis equal
hold off

%% Implement SVM

svm_classifier = fitcsvm(data3, theclass, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'BoxConstraint', 0.001, 'ClassNames', [-1, 1]);

%% Plot Decision Boundary

[~, scores] = predict(svm_classifier, xGrid);

hold on
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
ezpolar(@(x)1);
h(3) = plot(data3(svm_classifier.IsSupportVector,1),data3(svm_classifier.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'g');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off

%% Train/Evaluate Kernel Perceptron on Optdigits

clear
close
clc

load('optdigits49_test.txt');
load('optdigits49_train.txt');
load('optdigits79_test.txt');
load('optdigits79_train.txt');

[alpha, b] = kernPercGD(optdigits49_train(:,1:end-1),optdigits49_train(:,end));

for i = 1:size(optdigits49_test,1)
    
    for t=1:size(alpha,1)
        w_i(t) = alpha(t) .* optdigits49_train(t,end) .* (optdigits49_train(t,1:end-1) * optdigits49_test(i,1:end-1)').^2;
    end    
    
    if sum(w_i) + b > 0
        opt49_label(i,1) = 1;
    else
        opt49_label(i,1) = -1;
    end
    
end

error_rate = sum(opt49_label ~= optdigits49_test(:,end))./size(optdigits49_test,1);
sprintf('Error rate on test dataset: %.2f', error_rate * 100)

clearvars alpha b w_i

[alpha, b] = kernPercGD(optdigits79_train(:,1:end-1),optdigits79_train(:,end));

for i = 1:size(optdigits79_test,1)
    
    for t=1:size(alpha,1)
        w_i(t) = alpha(t) .* optdigits79_train(t,end) .* (optdigits79_train(t,1:end-1) * optdigits79_test(i,1:end-1)').^2;
    end    
    
    if sum(w_i) + b > 0
        opt79_label(i,1) = 1;
    else
        opt79_label(i,1) = -1;
    end
    
end

error_rate = sum(opt79_label ~= optdigits79_test(:,end))./size(optdigits79_test,1);
sprintf('Error rate on test dataset: %.2f', error_rate * 100)