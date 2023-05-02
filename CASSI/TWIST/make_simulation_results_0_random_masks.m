
%% Init
img = imread( './simulation_results/coded.png' );
output_dir = './simulation_results/0_extract_9_random_masks';
if ~exist( output_dir, 'dir' )
   mkdir( output_dir ); 
end
n_vertical_samples = 3;
n_horizontal_samples = 3;
[ c_height, c_width, c_channels ] = size( img );
target_height = 645;
target_width = 501;
pixel_ratio = 0.5;
intermediate_height = int32( target_height*pixel_ratio );
intermediate_width = int32( target_width*pixel_ratio );
vertical_grid_size = int32( c_height/n_vertical_samples );
horizontal_grid_size = int32( c_width/n_horizontal_samples );
padding = 120;
yellow = uint8([255 255 0]);
shapeInserter = vision.ShapeInserter( 'BorderColor','Custom',...
                                        'CustomBorderColor',yellow,...
                                        'LineWidth', 5 ) ;
img_to_display = repmat( img, 1, 1, 3 );


%% sample
for j = 1:n_vertical_samples
    start_y = ( j - 1 )*vertical_grid_size + padding;
    for i = 1:n_horizontal_samples
        %% Crop
        start_x = ( i - 1 )*vertical_grid_size + padding;
        rectangle...
            = int32([ start_y start_x intermediate_height intermediate_width ]);
        cropped_img = imcrop( img, rectangle );
        cropped_img...
            = imresize( cropped_img, [ target_width, target_height ], 'nearest' );
        %% Save
        filename = sprintf( 'cell_%d_%d.png', i, j );
        filename = fullfile( output_dir, filename );
        imwrite( cropped_img, filename );
        
        %% Draw
        img_to_display = step( shapeInserter, img_to_display, rectangle );
        imshow(img_to_display ); 
    end
end

