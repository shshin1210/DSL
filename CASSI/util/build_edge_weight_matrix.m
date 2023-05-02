function Wspatial = build_edge_weight_matrix(im_dispersed_rgb)

gaussian_sigma_value = 2;
gaussian_fsize = 7;


% data_path = 'Y:\papers\prism\scene\20170702_synthetic_scene4\100\';
% blurred_fn = 'blurred_rgb.png';
% im = im2double( imread([data_path blurred_fn]) );

% figure;
% imshow(im); title('observed');

im_grad = imgrad(im_dispersed_rgb, 'neumann');
im_grad_mag = zeros(size(im_dispersed_rgb));

for i = 1:3
    xgrad = im_grad(:,:,2*i-1);
    ygrad = im_grad(:,:,2*i);
    
    %     [im_grad_mag(:,:,i),~] = imgradient(xgrad, ygrad);
    [im_grad_mag(:,:,i),~] = imgradient(xgrad, ygrad);
    im_grad_mag(:,:,i) = abs(xgrad);
end

im_grad_mag_sum = sum(im_grad_mag, 3);
% figure; imagesc(im_grad_mag_sum);

im_grad_mag_sum = medfilt2(im_grad_mag_sum, [3, 3]);
% figure; imagesc(im_grad_mag_sum);

im_grad_mag_sum_filtered = imgaussfilt(im_grad_mag_sum, gaussian_sigma_value, 'Filtersize', gaussian_fsize );
% figure; imagesc(im_grad_mag_sum_filtered); colorbar;
% 
im_grad_mag_sum_filtered = im_grad_mag_sum_filtered/max(im_grad_mag_sum_filtered(:));


Wspatial = im_grad_mag_sum_filtered;


end
