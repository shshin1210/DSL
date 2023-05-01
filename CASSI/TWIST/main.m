clear;

%checkerboard = im2double(imread('checkerboard.pgm'));
%coded = im2double(imread('coded.pgm'));
%impts = im2double(imread('pts.pgm'));
%scene = im2double(imread('wheel01.pgm'));

center_col = 2;
center_row = 2;

[width, height] = size(checkerboard);

cell_width = 980;
cell_height = 720;

num_col = 3;
num_row = 3;

offset_x = 250;
offset_y = 266;

% mkdir('checkerboard_original_cells');
% mkdir('checkerboard_transformed_cells');
% mkdir('checkerboard_windowed_cells');
% mkdir('checkerboard_detected_pts');
% mkdir('checkerboard_matched_pts');
% mkdir('checkerboard_showpair');
% mkdir('checkerboard_showpair_before');
% mkdir('results');


%draw_cells;
%generate_registration_manual;
%mask_remove;

%% Apply homography to video frames
video_path = 'cube';
video_files = dir([video_path '/*.pgm']);

mkdir([video_path '_transform']);
for vfi=1:length(video_files)
    mkdir('scene_original_cells');
    mkdir('scene_transformed_cells');
    mkdir('scene_windowed_cells');
    
    video_file = video_files(vfi);
    
    scene = im2double(imread([video_path '/' video_file.name]));
    apply_homography;
    movefile('scene_windowed_cells', sprintf('data/%s/frame%04d', video_path, vfi));
end
cassivideo();
%{
mkdir('scene_original_cells');
mkdir('scene_transformed_cells');
mkdir('scene_windowed_cells');

dos(['rm ' '-rf ' 'scene']);
dos(['rm ' '-rf ' 'coded']);

apply_homography;
dos(['mv ' 'scene_windowed_cells ' 'scene']);
scene = coded;
apply_homography;
dos(['mv ' 'scene_windowed_cells ' 'coded']);
%}

%{
scene = im2double(imread('coded.pgm'));
G = fspecial('gaussian',[14 14],50);
for i=1:20
    scene = imfilter(scene,G,'same');
end
imshow(scene);
apply_homography;
%}

%{
files = dir('calib_transformed_cells/*.png'); 
for i=1:length(files)
    
end
%}
%close;