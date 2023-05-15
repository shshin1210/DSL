clear all;
close all;
warning off;

% image directory
img_path = "C:/Users/owner/Downloads/test_2023_05_11_16_00_crop/imgs";

file_list = dir(fullfile(img_path, '*.png'));

for i = 1:numel(file_list)
    % read image
    file_name = fullfile(img_path, file_list(i).name);
    img = imread(file_name);
    
    % rgb image to gray scale
    img = rgb2gray(img);       
    
    % extract to gray scale
    bw = img > 30;

    % extract index points
    s = regionprops(bw, 'Centroid');
    save_fn = fullfile(img_path, file_list(i).name(1:5) + "_centroid.mat");
    save(save_fn, "s")

    % visualization
    imshow(img)
    hold on
    for k = 1:numel(s)
        centroid_k = s(k).Centroid;
        plot(centroid_k(1), centroid_k(2), 'r.');
    end
    hold off

end