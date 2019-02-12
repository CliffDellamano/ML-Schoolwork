function [ p1, p2, pc1, pc2 ] = Bayes_Learning( training_data, validation_data )

load(training_data);

x_t = SPECT_train(:,1:end-1);
r_t = [SPECT_train(:,end) == 1, SPECT_train(:,end) == 2];

p = zeros(2, 22);

for i=1:2
    for j=1:22
        
        p(i,j) = sum(x_t(:,j).*r_t(:,i))./sum(r_t(:,i));
        
    end
end

load(validation_data);

x_v = SPECT_valid(:,1:end-1);
r_v = [SPECT_valid(:,end) == 1, SPECT_valid(:,end) == 2];

i = 1;

for sig = -5:5
  
    p_c1 = 1/(1 + exp(-sig));
    p_c2 = 1 - p_c1;
    
    r_pred = zeros(size(r_v));
    
    for t = 1:length(x_v)
        
        p_x_c1 = 1;
        p_x_c2 = 1;
        
        for j = 1:22 
            if x_v(t,j) == 1
                p_x_c1 = p_x_c1 * p(1,j);
                p_x_c2 = p_x_c2 * p(2,j);
            else
                p_x_c1 = p_x_c1 * (1 - p(1,j));
                p_x_c2 = p_x_c2 * (1 - p(2,j));
            end
        end
        
        if log((p_c1*p_x_c1)/(p_c2*p_x_c2)) > 0
            r_pred(t,1) = 1;
        else
            r_pred(t,2) = 1;
        end
    end
   
    num_correct(i) = sum(sum(r_v.*r_pred));
    error_rate(i) = (length(x_v)-num_correct(i))/(length(x_v)) * 100;
    
    if i > 1
        if num_correct(i) > max(num_correct(1:i-1))
            pc1 = p_c1;
            pc2 = p_c2;
        end
    end
    
    i = i + 1;
    
end

p1 = 1 - p(1,:);
p2 = 1 - p(2,:);

sprintf(' %.2f ', error_rate)

end

