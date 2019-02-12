function [w, step] = MyPerceptron(X, y, w0)

w = w0;
err = 1;
step = 0;

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
b = (-w0(1)*a)/w0(2);
plot(a,b,'k')
title('Initial Boundary')
axis([-1 1 -1 1])


while err > 0
    for i = 1:size(X,1)
        if dot(w,X(i,:))*y(i) <= 0
            w = w + y(i)*X(i);
        end
    end
    
    step = step + 1;
    err = sum(sign(X*w) ~= y) / (size(X,1));
end

figure;
scatter(X_blue(:,1), X_blue(:,2), 'b', 'filled')
hold on
scatter(X_red(:,1), X_red(:,2), 'r', 'filled')
b = (-w(1)*a)/w(2);
plot(a,b,'k')
title('After Perceptron Convergence')
axis([-1 1 -1 1])

end

