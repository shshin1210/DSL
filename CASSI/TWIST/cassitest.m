function cassitest( path, output_path )

if nargin < 2
    output_path = '/projected/';
end

nt = 5;
w = 512;
h = 512;
flipped = [1 5];

wvls2b = 400:10:700;

%% Dispersion
disperse_csv = csvread('data/dispersed.csv');
wvls = disperse_csv(:, 1);
disperse = disperse_csv(:, 2);
fittedmodel = fit(wvls, disperse, 'power2');
ref_disperse = fittedmodel(636);
disperse = fittedmodel(wvls2b) - ref_disperse;
disperse = int8(disperse);


%% Coded aperture
mkdir('cells_coded_warped');
full_mask = imread('coded.png');

for i=1:nt
    c = mod(i, 3);
    r = (i / 3);
    
    x = floor(c * (w) + 1);
    y = floor(r * (h) + 1);
    
    coded(:,:,i) = imresize(im2double(imcrop(full_mask, [x, y, w-1, h-1])), [w, h]);
end

%coded(:,:,:) = 1;

for i=1:nt
    imwrite(coded(:,:,i), sprintf('cells_coded_warped/cell_%d.png', i));
end

%% Read spectrum files
files = dir(sprintf('%s/*.png', path));

if length(files) == 0
    error(['No files in ' path]);
end

for i = 1:length(files)
    filenames{i} = files(i).name;
end
filenames = sort_nat(filenames);

projected = zeros([h, w, nt]);
Cu = zeros([h, w, size(wvls2b, 2)]);

for i = 1:length(filenames)
    filename = filenames{i};
    
    band = im2double(imread(sprintf('%s/%s', path, filename)));
    band = imresize(band, [w, h]);
    
    Cu(:,:,i) = band;
    % ref(:,:,i) = band;
    
    for j = 1:size(coded,3)
        coded_band = coded(:,:,j) .* band;
        if ismember(j, flipped)
            projected(:,:,j) = projected(:,:,j) + circshift(coded_band, [0, -disperse(i)]);
        else 
            projected(:,:,j) = projected(:,:,j) + circshift(coded_band, [0, disperse(i)]);
        end
    end
end

save(sprintf('%s/gt.mat', path), 'Cu');

%% Wrtie projected images
scene_path = sprintf('%s/%s/', path, output_path);
mkdir(scene_path);
len_wvls = length(wvls2b);
for i=1:nt
    imwrite(projected(:,:,i) / len_wvls, sprintf('%s/%s/cell_%d.png', path, output_path, i));
end


%% Reconstruction
root_path = '.';
cassirecon_sj(root_path, scene_path, [1 1 w-1 h-1], wvls2b, flipped);

end

