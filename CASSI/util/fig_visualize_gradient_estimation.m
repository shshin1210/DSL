data_path = 'Y:\papers\prism\scene\20170522_goose2\results_ic_1\';

edge_map = imread([data_path '2_edge\mask.png']);

lambdas = 420:10:680;
n_lambdas = numel(lambdas);

grad_fn = 'recon_result_step3.mat';

load([data_path grad_fn]);

% plot the gradient plot
figure; imagesc(sum(abs(im_grad_spec),3));
[im_crop, im_crop_rect] = imcrop(sum(abs(im_grad_spec),3));
im_crop_rect = round(im_crop_rect);
im_grad_vec = im_grad_spec(im_crop_rect(2):(im_crop_rect(2)+im_crop_rect(4)), im_crop_rect(1):(im_crop_rect(1)+im_crop_rect(3)), :);
edge_map_vec = edge_map(im_crop_rect(2):(im_crop_rect(2)+im_crop_rect(4)), im_crop_rect(1):(im_crop_rect(1)+im_crop_rect(3)));

mean_grad_values = squeeze(sum(sum(im_grad_vec,1),2)) / sum(edge_map_vec(:));
mean_grad_values_hor = mean_grad_values(1:2:end);
mean_grad_values_ver = mean_grad_values(2:2:end);
figure; plot(lambdas, mean_grad_values_hor);
figure; plot(lambdas, mean_grad_values_ver);

figure; imagesc(im_grad_spec(:,:,29), [-0.2, 0.2]); axis off; colormap(othercolor('RdBu11')); colorbar; axis equal;
