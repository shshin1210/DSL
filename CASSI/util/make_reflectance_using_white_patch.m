function msimg_reflectance = make_reflectance_using_white_patch(msimg_radiance, lambdas)

params.lambdas = lambdas;
params.illuminant = 'd65';
params.cmf = '1931_full';

im_srgb = spec2srgb(msimg_radiance, params);
figure; 
title('Select a white patch');
hold on;
[w_patch_srgb, w_patch_rect] = imcrop(im_srgb); 
hold off;
close(gcf);

w_patch_spec = squeeze(mean(mean(msimg_radiance(w_patch_rect(2):w_patch_rect(2)+w_patch_rect(4)-1, w_patch_rect(1):w_patch_rect(1)+w_patch_rect(3)-1, :), 1), 2));

[N,M,L] = size(msimg_radiance);

msimg_reflectance = bsxfun(@rdivide, reshape(msimg_radiance, [], L),  w_patch_spec') * 0.7;%0.99;
% msimg_reflectance = bsxfun(@rdivide, reshape(msimg_radiance, [], L),  w_patch_spec') * ;

msimg_reflectance = reshape(msimg_reflectance, N, M, L);

im_srgb_reflectance = spec2srgb(msimg_reflectance, params);

figure; 
imshow(im_srgb_reflectance);
title('reflectance image');

end