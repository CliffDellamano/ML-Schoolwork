%% Blank Workspace

clear
close
clc

%% Load Training Dataset

load('SPECT_train.txt');

x = SPECT_train(:,1:end-1);
r = [SPECT_train(:,end) == 1, SPECT_train(:,end) == 2];

%% Learn Parameters

p = zeros(2, 22);

for i=1:2
    for j=1:22
        
        p(i,j) = sum(x(:,j).*r(:,i))./sum(r(:,i));
        
    end
end

%% Load Validation Dataset

load('SPECT_valid.txt');

x_v = SPECT_valid(:,1:end-1);
r_v = [SPECT_valid(:,end) == 1, SPECT_valid(:,end) == 2];

%% Classify With Priors

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
    i = i + 1;
    
end

%% Load Test Data

load('SPECT_test.txt');

x_t = SPECT_test(:,1:end-1);
r_t = [SPECT_test(:,end) == 1, SPECT_test(:,end) == 2];

%% Classify Test Data

p_c1 = 1/(1 + exp(-(-4)));
p_c2 = 1 - p_c1;

r_pred = zeros(size(r_t));

for t = 1:length(x_t)

    p_x_c1 = 1;
    p_x_c2 = 1;

    for j = 1:22 
        if x_t(t,j) == 1
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

num_correct = sum(sum(r_t.*r_pred));

error_rate = ((length(x_t) - num_correct)/length(x_t)) * 100;


    