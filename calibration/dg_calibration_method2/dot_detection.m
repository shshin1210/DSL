clear all;
close all;
warning off;

% image directory
% date = "test_2023_07_09_15_37_(2)";
date = "20230822_data/back";
test_fn = date + "_processed";
% test_fn = date;

img_test_path = "C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/calibration/dg_calibration_method2/";

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
        % gamma
%         img = histeq(img);
        img = imadjust(img,[],[],0.5);

        minradius = 2;
        maxradius = 10;
        [centers, radii, metric] = imfindcircles(img, [minradius, maxradius], 'EdgeThreshold', 0.027);
        
        % visualization
        figure(1);
        imshow(img)
        hold on

        for k = 1:size(centers,1)
            plot(centers(k,1), centers(k,2), 'g.');
        end

        titleString = [pattern_file_list(i).name,' ', wvls_file_list(j).name(1:5),' ',num2str(size(centers,1))];
        title(titleString);

        hold off
%         pause(0.5);
        pause(0.1);
        
%         cmd_c = input('type 0 and enter to correct the points:');
%         if cmd_c == 0
%             figure;
%             imagesc(img);
%             N = input('type number of points:');
%             fprintf('click points one by one\n');
%             [xi, yi] = ginput(N);
%             
%             centers_re = [];
%             for k = 1:N
%                 img_cur = zeros(size(img));
%                 ymin = yi(k)-maxradius;
%                 ymax = yi(k)+maxradius;
%                 xmin = xi(k)-maxradius;
%                 xmax = xi(k)+maxradius;
%                 
%                 if ymin < 0
%                     ymin = 1;
%                 end
% 
%                 if xmin < 0
%                     xmin = 1;
%                 end
% 
%                 if ymax > 580
%                     ymax = 579;
%                 end
%                 
%                 if xmax > 890
%                     xmax = 889;
%                 end
% 
%                 img_cur(ymin:ymax, xmin:xmax,:) = img(ymin:ymax, xmin:xmax,:);
%                 [centers_k, radii, metric] = imfindcircles(img_cur, [minradius, maxradius], 'EdgeThreshold', 0.027);
%                 centers_re = [centers_re; centers_k(1,:)];
% 
%             end
%             figure(1);
%             imshow(img)
%             hold on
% 
%             for k = 1:size(centers_re,1)
%                 plot(centers_re(k,1), centers_re(k,2), 'g.');
%             end
%             title(size(centers_re,1));
%             centers = centers_re;
%         end


        % save points in new folder
        save_fn = fullfile(img_test_path, test_points_fn,  pattern_file_list(i).name);
        
        if ~exist(save_fn, 'dir')
            mkdir(save_fn)
        end

        mat_file = fullfile(save_fn, wvls_file_list(j).name(1:5) + "_centroid.mat");

        save(mat_file, "centers")

    end
end
