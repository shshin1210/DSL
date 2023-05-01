function display_spectral_images_graph(im_spec)

grid_col = 7;

[N,M,nCh] = size(im_spec);
grid_row = ceil(nCh/grid_col);

%fh = figure(300);
fh = figure;
set(fh, 'Position', [100, 100, 1500, 1500 * (grid_row/grid_col)]);

ind = 1;
for r = 1:grid_row
    for c = 1:grid_col
        if ind > nCh
            break;
        end
        subplot(grid_row,grid_col,ind); imshow(im_spec(:,:,ind)*1);
        ind = ind + 1;
    end
    if ind > nCh
        break;
    end
end
drawnow;
im_spec_vec = squeeze( mean(mean(im_spec)) );
subplot(grid_row,grid_col,ind); plot(im_spec_vec, '--o');

end