load('homographys.mat');
%load('homographys2.mat');
load('opticalflows.mat');

mkdir('scene_original_cells');
mkdir('scene_transformed_cells');
mkdir('scene_windowed_cells');

% Fixed cell (center cell)
x = cell_width * (center_col - 1) + offset_x;
y = cell_height * (center_row - 1) + offset_y;
fixed_cell = imcrop(scene, [x y cell_width-1 cell_height-1]);

% Save original scene cell
imwrite(fixed_cell, sprintf('scene_original_cells/cell_%d_%d.png', center_col, center_row));
imwrite(fixed_cell, sprintf('scene_transformed_cells/cell_%d_%d.png', center_col, center_row));
        
cell_index = center_row + (center_col - 1) * num_row;
%total(:,:,cell_index) = fixed_cell;

%% Average of all images
total = fixed_cell;

outputView = imref2d(size(fixed_cell));

for col=1:num_col
    for row=1:num_row
        if col == center_col && row == center_row
            continue;
        end
        
        if mod(col + row, 2) == 0
            continue;
        end
        
        cell_index = row + (col - 1) * num_row;
        
        % Get homography matrix
        tform = homographys(cell_index);
        
        if ~use_for_sr(cell_index)
            continue;
        end
        
        x = cell_width * (col - 1) + offset_x;
        y = cell_height * (row - 1) + offset_y;
        
        cell = imcrop(scene, [x y cell_width-1 cell_height-1]);
        
        % Save original scene cell
        imwrite(cell, sprintf('scene_original_cells/cell_%d_%d.png', col, row));
        
        % Warping
        cell = imwarp(cell, tform,'OutputView',outputView);
        %cell = imwarp(cell, homographys2(cell_index),'OutputView',outputView);
        
        
        V = opticalflows{cell_index};
        vx = V(:,:,1);
        vy = V(:,:,2);
        cell = warpImage(cell,vx,vy);
        
        % Save transformed scene cell
        imwrite(cell, sprintf('scene_transformed_cells/cell_%d_%d.png', col, row));
        
        %total(:,:,cell_index) = cell;
        total = cat(3, total, cell);
    end
end
%average = mean(total, 3);
%med = median(total, 3);

%imwrite(fixed_cell, 'results/original.png');
%imwrite(average, 'results/average.png');
%imwrite(med, 'results/median.png');

%imshow(average);


window = [70 50 840 640];

files = dir('scene_transformed_cells');
for i=1:length(files)
    file = files(i);
    if file.isdir
        continue
    end
    im = imread(['scene_transformed_cells/' file.name]);
    imwrite(imcrop(im, window), ['scene_windowed_cells/' file.name]);
end