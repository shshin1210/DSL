function cassimatvideo(path, band_index, low, high)

mkdir([path '/rgb']);
mkdir([path '/band']);
mkdir([path '/video']);

load([path '/0010.mat']);
    
% Write Video
srgb_writer = VideoWriter(sprintf('%s/video/srgb.avi', path));
srgb_writer.FrameRate = 10; 
open(srgb_writer);

band_writer = VideoWriter(sprintf('%s/video/band.avi', path));
band_writer.FrameRate = 10; 
open(band_writer);

spectrum_writer = VideoWriter(sprintf('%s/video/spectrum.avi', path));
spectrum_writer.FrameRate = 10; 
open(spectrum_writer);

refwhite = [0.0334488
0.0371973
0.03864
0.037611
0.037086
0.0361977
0.0347949
0.0328944
0.0325164
0.0322035
0.0329826
0.0326991
0.0332724
0.0350007
0.0341334
0.0342573
0.0352779
0.0381696
0.038556
0.0385791
0.0396858
0.04158
0.036666
0.0343833
0.034419
0.0317814
0.0278523
0.0264012
0.0264012] / 32;

offset_x = 50;
offset_y = 50;
paste_w = 240;
paste_h = 168;
space_x = 20;
space_y = 77;


matfiles = dir2([path '/*.mat']);

for matfile = matfiles'
    
    load([path '/' matfile.name]);
%     
%     for i=1:size(wvls2b, 2)
%         x_calib(:,:,i) = polyval(fitted(i,:), x_recon(:,:,i));
%     end
    [pathstr,name,ext] = fileparts(matfile.name);

    im_rgb = R2sRGB(wvls2b,x_recon,refwhite, 0, 0);
    
    im_rgb = stretch_hist(im_rgb, low, high);
    
%     im_rgb = im_rgb * max(x_recon(:)) / ref_value;
    im_rgb(im_rgb > 1) = 1;
    im_rgb(im_rgb < 0) = 0;
    
    im_band = x_recon(:,:,band_index) / refwhite(band_index) / 32;
    im_band(im_band > 1) = 1;
    im_band(im_band < 0) = 0;
    
    imwrite(im_rgb, sprintf('%s/rgb/%s.png', path, name));
    imwrite(im_band, sprintf('%s/band/%s.png', path, name));
    
%     save(sprintf('%s/result/mat/%s.mat', scene_root_path, scene_path{1}), 'x_recon', 'wvls2b');
   

    writeVideo(srgb_writer, im_rgb);
    writeVideo(band_writer, im_band);
    
    spectrum_frame = zeros(1080, 1920, 3);
    
    wvl_index = 28;
    im_x = offset_x + mod(wvl_index-1, 7) * (paste_w + space_x);
    im_y = offset_y + floor((wvl_index-1) / 7) * (paste_h + space_y); 
    
    frame = imresize(im_rgb, [paste_h, paste_w]);
    
    spectrum_frame(im_y:im_y+paste_h-1,im_x:im_x+paste_w-1,:) = frame;
    spectrum_frame = insertText(spectrum_frame, [im_x+(paste_w/2) im_y+paste_h+24],'sRGB', ...
        'TextColor', 'white', ...
        'AnchorPoint', 'Center', ...
        'BoxOpacity', 0, ...
        'FontSize', 24);
    
    resized = imresize(x_recon, [paste_h, paste_w]);
    resized = max(resized, 0);
    resized = min(resized, 1);
    
    wvls = 450:10:700;
    for i=1:length(wvls)
        wvl = wvls(i);
        
        wvl_index = i;
        im_x = offset_x + mod(wvl_index-1, 7) * (paste_w + space_x);
        im_y = offset_y + floor((wvl_index-1) / 7) * (paste_h + space_y); 
        
        rgb = spectrumRGB(wvl);
        spectrum_mono = resized(:,:,i+3);
        frame(:,:,1) = spectrum_mono / refwhite(i+3) / 32 * rgb(1);
        frame(:,:,2) = spectrum_mono / refwhite(i+3) / 32 * rgb(2);
        frame(:,:,3) = spectrum_mono / refwhite(i+3) / 32 * rgb(3);
        
        
        spectrum_frame(im_y:im_y+paste_h-1,im_x:im_x+paste_w-1,:) = frame;
        spectrum_frame = insertText(spectrum_frame, [im_x+(paste_w/2) im_y+paste_h+24],sprintf('%dnm', wvl), ...
            'TextColor', 'white', ...
            'AnchorPoint', 'Center', ...
            'BoxOpacity', 0, ...
            'FontSize', 24);
    end
    
    spectrum_frame(spectrum_frame > 1) = 1;
    spectrum_frame(spectrum_frame < 0) = 0;
    
    writeVideo(spectrum_writer, spectrum_frame);
end

close(srgb_writer);
close(band_writer);
close(spectrum_writer);


% Write Video
scene_writer = VideoWriter(sprintf('%s/video/scene.avi', path));
scene_writer.FrameRate = 10; 
open(scene_writer);

scene_files = dir([path '/scene/*.png']);

for scene_file=scene_files'
    im_scene = im2single(imread([path '/scene/' scene_file.name]));
    writeVideo(scene_writer, im_scene);
end

close(scene_writer)

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
