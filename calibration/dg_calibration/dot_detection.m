clear all;
close all;
warning off;

% image directory
test_fn = "test_2023_05_15_17_34_processed";
img_test_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/";

% save points directory
test_points_fn = "test_2023_05_15_17_34_points";

img_path = append(img_test_path, test_fn);

wvl_file_list = dir(img_path);
wvl_file_list = wvl_file_list(~ismember({wvl_file_list(:).name},{'.','..'}));

for i = 1:numel(wvl_file_list)
    % into wvl dirs
    wvl_fn = fullfile(img_path, wvl_file_list(i).name);
    
    % into pattern dirs
    pattern_file_list = dir(wvl_fn);
    pattern_file_list = pattern_file_list(~ismember({pattern_file_list(:).name}, {'.','..'}));
    
    for j = 1:numel(pattern_file_list)
        pattern_fn = fullfile(img_path, wvl_file_list(i).name, pattern_file_list(j).name);
        
        % into captured files
        captured_file_list = dir(pattern_fn);
        captured_file_list = captured_file_list(~ismember({captured_file_list(:).name}, {'.','..'}));
        fn = fullfile(img_path, wvl_file_list(i).name, pattern_file_list(j).name, captured_file_list.name);

        img = imread(fn);
        
        % rgb image to gray scale
        img = rgb2gray(img); 

        % extract to gray scale
        bw = img > 30;

        % extract index points
        s = regionprops(bw, 'Centroid');
        
        % visualization
        imshow(img)
        hold on
        for k = 1:numel(s)
            centroid_k = s(k).Centroid;
            plot(centroid_k(1), centroid_k(2), 'r.');
        end
        hold off
        
        % save points in new folder
        save_fn = fullfile(img_test_path, test_points_fn, wvl_file_list(i).name, pattern_file_list(j).name);
        
        if ~exist(save_fn, 'dir')
            mkdir(save_fn)
        end

        mat_file = fullfile(save_fn, captured_file_list.name(1:12) + "_centroid.mat");

        save(mat_file, "s")
        
    end

end