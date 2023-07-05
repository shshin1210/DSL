clear all;
close all;
warning off;

% image directory
date = "test_2023_07_04_15_41";
test_fn = date + "_processed";
img_test_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration/";

% save points directory
test_points_fn = date + "_points";
% image directory
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
        img = medfilt2(img, [3,3]);

        % extract from gray scale
%         bw = img > 28; % 31

        % extract index points
        %s = regionprops(bw, 'Centroid');
%         s = regionprops(img, 'Centroid');
%         s = regionprops(img);
        minradius = 2;
        maxradius = 10;
        [centers, radii, metric] = imfindcircles(img, [minradius, maxradius], 'EdgeThreshold', 0.027);
%         
%         centers= rmoutliers(centers);
        
        % visualization
        figure(1);
        imshow(img)
        hold on
%         for k = 1:numel(s)
%             centroid_k = s(k).Centroid;
%             plot(centroid_k(1), centroid_k(2), 'r.');
%         end
%         for k = 1:size(centroid_k_inliers,1)
%             plot(centroid_k_inliers(k,1), centroid_k_inliers(k,2), 'g.');
%         end


        for k = 1:size(centers,1)
            plot(centers(k,1), centers(k,2), 'g.');
        end
        title(size(centers,1));

        hold off
        pause(0.5);

        cmd_c = input('type 0 and enter to correct the points:');
        if cmd_c == 0
            figure;
            imagesc(img);
            N = input('type number of points:');
            fprintf('click points one by one\n');
            [xi, yi] = ginput(N);
            
            centers_re = [];
            for k = 1:N
                img_cur = zeros(size(img));
                ymin = yi(k)-maxradius;
                ymax = yi(k)+maxradius;
                xmin = xi(k)-maxradius;
                xmax = xi(k)+maxradius;
                img_cur(ymin:ymax, xmin:xmax,:) = img(ymin:ymax, xmin:xmax,:);
                [centers_k, radii, metric] = imfindcircles(img_cur, [minradius, maxradius], 'EdgeThreshold', 0.027);
                centers_re = [centers_re; centers_k(1,:)];
            end
            figure(1);
            imshow(img)
            hold on

            for k = 1:size(centers_re,1)
                plot(centers_re(k,1), centers_re(k,2), 'g.');
            end
            title(size(centers_re,1));
            centers = centers_re;
        end


        % save points in new folder
        save_fn = fullfile(img_test_path, test_points_fn,  pattern_file_list(i).name);
        
        if ~exist(save_fn, 'dir')
            mkdir(save_fn)
        end

        mat_file = fullfile(save_fn, wvls_file_list(j).name(1:5) + "_centroid.mat");

%         save(mat_file, "s")
        save(mat_file, "centers")

    end
end
