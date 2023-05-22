clear all;
close all;
warning off;

% image directory
test_fn = "test_2023_05_15_17_34_processed";
img_test_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/";

% save points directory
test_points_fn = "test_2023_05_15_17_34_points";

img_path = append(img_test_path, test_fn);

pattern_file_list = dir(img_path);
pattern_file_list = pattern_file_list(~ismember({pattern_file_list(:).name},{'.','..'}));

for i = 1:numel(pattern_file_list)
    % into pattern dirs
    pattern_fn = fullfile(img_path, pattern_file_list(i).name);
    
    % into wvl files
    wvls_file_list = dir(pattern_fn);
    wvls_file_list = wvls_file_list(~ismember({wvls_file_list(:).name}, {'.','..'}));
    
    for j = 1:numel(wvls_file_list)
        img_fn = fullfile(img_path, pattern_file_list(i).name, wvls_file_list(j).name);
        
        % read image
        img = imread(img_fn);
        
        % rgb image to gray scale
        img = rgb2gray(img); 

        % extract from gray scale
        bw = img > 30;

        % extract index points
        s = regionprops(bw, 'Centroid');
        
        % visualization
        figure(1);
        imshow(img)
        hold on
        for k = 1:numel(s)
            centroid_k = s(k).Centroid;
            plot(centroid_k(1), centroid_k(2), 'r.');
        end
        hold off
        pause(1.5);

        % save points in new folder
        save_fn = fullfile(img_test_path, test_points_fn,  pattern_file_list(i).name);
        W
        if ~exist(save_fn, 'dir')
            mkdir(save_fn)
        end

        mat_file = fullfile(save_fn, wvls_file_list(j).name(1:15) + "_centroid.mat");

        save(mat_file, "s")

    end
end
