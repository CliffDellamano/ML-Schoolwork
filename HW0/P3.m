%% Clear Workspace

clear
close
clc

%% Part 1

load('data1.mat');
[w, step] = MyPerceptron(X, y, [1;-1]);

%% Part 2

load('data2.mat');
[m,n]=size(X);
f=[zeros(n,1);ones(m,1)]; % transform problem into a standard LP
A1=[X.*repmat(y,1,n),eye(m,m)];
A2=[zeros(m,n),eye(m,m)];
A=-[A1;A2];
b=[-ones(m,1);zeros(m,1)];
x = linprog(f,A,b);% solve LP
w=x(1:n);% return varible w

X_blue = [];
X_red = [];
for n=1:length(y)
    if y(n)==1
        X_blue = [ X_blue; X(n,:) ];
    elseif y(n) == -1
        X_red = [ X_red; X(n,:) ];
    end
end

figure;
scatter(X_blue(:,1), X_blue(:,2), 'b', 'filled')
hold on
scatter(X_red(:,1), X_red(:,2), 'r', 'filled')
a = linspace(-1,1,10);
b = (-w(1)*a)/w(2);
plot(a,b,'k')
title('After "Soft Classifier" Convergence')
axis([-1 1 -1 1])
