function [  ] = Bayes_Testing( test_data, p1, p2, pc1, pc2 )

load(test_data);

x_t = SPECT_test(:,1:end-1);
r_t = [SPECT_test(:,end) == 1, SPECT_test(:,end) == 2];

r_pred = zeros(size(r_t));

for t = 1:length(x_t)

    p_x_c1 = 1;
    p_x_c2 = 1;

    for j = 1:length(p1) 
        if x_t(t,j) == 1
            p_x_c1 = p_x_c1 * (1 - p1(j));
            p_x_c2 = p_x_c2 * (1 - p2(j));
        else
            p_x_c1 = p_x_c1 * p1(j);
            p_x_c2 = p_x_c2 * p2(j);
        end
    end

    if log((pc1*p_x_c1)/(pc2*p_x_c2)) > 0
        r_pred(t,1) = 1;
    else
        r_pred(t,2) = 1;
    end
end

num_correct = sum(sum(r_t.*r_pred));
error_rate = ((length(x_t) - num_correct)/length(x_t)) * 100;

sprintf('Error rate with best prior: %.2f', error_rate)

end

