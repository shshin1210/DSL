function cassirecon_batch( root_path )
%CASSIRECON_BATCH Summary of this function goes here
%   Detailed explanation goes here


scene_root_path = sprintf('%s/scene_warped/', root_path);
scene_dirs = dir2(scene_root_path);

%% Crop Region
scene_dir = scene_dirs(1);
scene_files = dir([scene_root_path '/' scene_dir.name '/*.png']);
im = imread([scene_root_path '/' scene_dir.name '/' scene_files(1).name]);
crect = [1 1 size(im, 2)-1 size(im, 1)-1];

%% Iterate
for scene_dir = scene_dirs'    
    cassirecon_sj(root_path, [scene_root_path '/' scene_dir.name '/'], crect);
end

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
    if any(strcmp(listing(k).name, {'.', '..'}))
        inds(end + 1) = k;
        n = n + 1;
    end
    k = k + 1;
end

listing(inds) = [];
end
