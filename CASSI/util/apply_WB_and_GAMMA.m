function [im_RGB_result,wb_rgb_factors] = apply_WB_and_GAMMA(im_RGB, gamma_val, wb_rgb_factors)

if ~exist('gamma_val', 'var')
    gamma_val = 2.2;
end

if ~exist('wb_rgb_factors', 'var')
    wb_rgb_factors = [];
end


[im_RGB] = im2double(im_RGB);
if isempty(wb_rgb_factors)
    
    figure; imshow(im_RGB);
    title('select a reference white patch');
    rect = getrect(gcf);
    
    min_r = rect(2);
    max_r = rect(2) + rect(4) - 1;
    min_c = rect(1);
    max_c = rect(1) + rect(3) - 1;
    
    white_RGB = squeeze(mean(mean(im_RGB(min_r:max_r, min_c:max_c, :),1),2));
    wb_rgb_factors = [(white_RGB(1)/white_RGB(2)), 1, (white_RGB(3)/white_RGB(2))];
end

im_RGB_result = zeros(size(im_RGB));

for i = 1:3
    im_RGB_result(:,:,i) = im_RGB(:,:,i) / wb_rgb_factors(i);
end

im_RGB_result = im_RGB_result.^(1/gamma_val);

im_RGB_result = min(im_RGB_result, 1);
im_RGB_result = max(im_RGB_result, 0);

end