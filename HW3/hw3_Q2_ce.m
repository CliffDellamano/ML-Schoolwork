%% Start with Blank Workspace

clear
close
clc

%% c.) Compare Standard EM and K-Means

%Use Standard EM
try 
    [h, m, Q] = EMG(0, 'goldy.bmp', 7); 
catch
    sprintf(['The above EM implementation failed because the covariance matrix ' ...
        'became singular. This is likely due to the fact that there are few ' ...
        'distinct colors in the goldy image, so some clusters have very few data ' ...
        'points in them and the variances of these clusters are near zero.'])
end

%Use K-Means

[img_g, cmap_g] = imread('goldy.bmp');
img_g_rgb = ind2rgb(img_g, cmap_g);
goldy = im2double(img_g_rgb);

goldy_data = reshape(goldy, size(goldy, 1)*size(goldy, 2), 3);
[IDX, C] = kmeans(goldy_data, 7);

for t = 1:size(IDX,1)
    goldy_data(t,:) = C(IDX(t),:);
end

goldy = reshape(goldy_data, size(goldy, 1), size(goldy, 2), size(goldy, 3));
imshow(goldy)

%% e.) Implement Improved EM

[h, m, Q] = EMG(1, 'goldy.bmp', 7); 

