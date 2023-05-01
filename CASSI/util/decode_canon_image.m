
path_to_dcraw = '..\dcraw\' ;
path_to_images = 'Y:\papers\prism\scene\20170414_spectrum_recon_zenon\';
full_path_to_images = [path_to_images '*.CR2'];
fprintf('linearization\n');
% to linear 16bit color image without WB
command = [path_to_dcraw, 'dcraw.exe -j -o 0 -4 -r 1 1 1 1 ', full_path_to_images];
% command = [path_to_dcraw, 'dcraw.exe -j -o 1 ', full_path_to_images];

status = dos(command);

load('Y:\papers\prism\calib\20170403_calib2\camera\cam_calib.mat');
fns = dir([path_to_images '*.ppm']);
fprintf('undistortion\n');
for i = 1:size(fns,1)
    im = imread([path_to_images fns(i).name]);
    uim = undistortImage(im, cameraParams, 'OutputView', 'same');
    imwrite(uim, [path_to_images 'u_' fns(i).name]); 
end
