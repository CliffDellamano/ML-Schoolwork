function [h, m, Q] = EMG(flag, image, k)

% Load and reshape image
[img cmap] = imread(image);
img_rgb = ind2rgb(img, cmap);
img_double = im2double(img_rgb);

clear img
img = reshape(img_double, size(img_double,1)*size(img_double,2), 3);

% Begin with k randomly generated means and st. devs for k clusters
mu = rand(k,3);

sigma = rand(3,3,k);
for i = 1:k
    sigma(:,:,i) = tril(sigma(:,:,i)) * tril(sigma(:,:,i))';
end

% Begin with k estimates for "pi", starting with equal probability for each
% cluster
pi = ones(k,1) * (1./k);

% Begin convergence loop
Q = 0;
iteration_diff = 100;
num_iterations = 0;

while iteration_diff >= 50 && num_iterations < 100
    
    % Perform "E-Step"
    gam_denom = zeros(size(img,1),1);
    
    for j = 1:k
        gam_temp = pi(j) .* mvnpdf(img, mu(j,:), sigma(:,:,j));
        gam_denom = gam_denom + gam_temp;
    end
    
    for i = 1:k
        gam_z(:,i) = (pi(i) .* mvnpdf(img, mu(i,:), sigma(:,:,i)))./(gam_denom);
    end
    
    % Perform "M-Step"
    Ni = sum(gam_z,1);
    
    pi = Ni./(length(img));
    
    for i = 1:k
        
        mu_temp = zeros(1,3);
        sig_temp = zeros(3,3);
        
        for t = 1:size(img,1)
            mu_temp_t = gam_z(t,i) .* img(t,:);
            mu_temp = mu_temp + mu_temp_t;
            
            if flag == 0
                sig_temp_t = gam_z(t,i) .* ((img(t,:) - mu(i,:))' * (img(t,:) - mu(i,:)));
            elseif flag == 1
                sig_temp_t = gam_z(t,i) .* (((img(t,:) - mu(i,:))' * (img(t,:) - mu(i,:))) + (10e-10 .* eye(3)));
            end
            sig_temp = sig_temp + sig_temp_t;
        end
        
        mu(i,:) = (1./Ni(i)) .* mu_temp;
        sigma(:,:,i) = (1./Ni(i)) .* sig_temp;
    end
    
    % Calculate improvement over previous iteration
    Q_curr = 0;
    
    for t = 1:size(img,1)
        for i = 1:k
            Q_temp = gam_z(t,i) .* (log(pi(i)) + log(mvnpdf(img(t,:), mu(i,:), sigma(:,:,i)))+10e-10);
            if ~isinf(Q_temp) && ~isnan(Q_temp)
                Q_curr = Q_curr + Q_temp;
            end
        end
    end
            
    Q = [Q; Q_curr];
    iteration_diff = abs(Q(end)-Q(end-1));
    num_iterations = num_iterations + 1;
    
end

% Outputs
plot(Q(2:end))

h = gam_z;
m = mu;
Q = Q';

%Assigning colors and compressing
for t = 1:size(img,1)
    [~,I] = max(gam_z(t,:));
    img(t,:) = mu(I,:);
end

img_compress = reshape(img, size(img_double,1), size(img_double,2), size(img_double,3));
figure
imshow(img_compress)

end

