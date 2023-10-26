clear all;
close all;
warning off;

dat_path = 'C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/dat';
% pitch =  7.9559e-06;

file_list = dir(fullfile(dat_path, 'depth_*.mat' ));

for i = 1:numel(file_list)
    file_name = fullfile(dat_path, file_list(i).name);
    load(file_name);
    
    output = createFit(x, y);

    file_name_out = fullfile(dat_path, sprintf('param_%s.mat', file_list(i).name(1:end-4)));

    p = [output.a, output.b, output.c];

    save(file_name_out, 'p')
end