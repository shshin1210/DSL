function cassivideo(root_path, scene_name)

% load('spectral_calib.mat');

scene_root_path = sprintf('%s/%s/', root_path, scene_name);
scene_root_path = '/home/sjjeon/Code/hyperspectralcam/data/20151229_video/video/';

if exist([scene_root_path '/result'])
    rmdir([scene_root_path '/result'], 's');
end

%% Iterate
scene_dirs = dir2(scene_root_path);
for i = 1:length(scene_dirs)
    scene_paths{i} = scene_dirs(i).name;
end
scene_paths = sort_nat(scene_paths);

% Make directory
mkdir([scene_root_path '/result']);
mkdir([scene_root_path '/result/rgb']);
mkdir([scene_root_path '/result/mat']);

% Write Video
writer = VideoWriter(sprintf('%s/result/output.avi', scene_root_path));
writer.FrameRate = 10; 
open(writer);

for scene_path = scene_paths    
    result_path = sprintf('%s/%s/result', scene_root_path, scene_path{1});
    
%     im_rgb = imread([result_path '/rgb.png']);

    matfiles = dir([result_path '/*.mat']);
    
    load([result_path '/' matfiles(1).name]);
%     
%     for i=1:size(wvls2b, 2)
%         x_calib(:,:,i) = polyval(fitted(i,:), x_recon(:,:,i));
%     end
    im_rgb = R2sRGB(wvls2b,x_recon,0, 0, 0);
    
    imwrite(im_rgb, sprintf('%s/result/rgb/%s.png', scene_root_path, scene_path{1}));
    
    save(sprintf('%s/result/mat/%s.mat', scene_root_path, scene_path{1}), 'x_recon', 'wvls2b');
    
    writeVideo(writer, im_rgb);
end

close(writer);

end



function listing = dir2(varargin)

if nargin == 0
    name = '.';
elseif nargin == 1
    name = varargin{1};
else
    error('Too many input arguments.')
end

listing = dir(name);

inds = [];
n    = 0;
k    = 1;

while n < 2 && k <= length(listing)
    if any(strcmp(listing(k).name, {'.', '..', 'result'}))
        inds(end + 1) = k;
        n = n + 1;
    end
    k = k + 1;
end

listing(inds) = [];
end
