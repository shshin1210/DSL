function [im_spec_color_coded,im_spec_color_coded_concat] = apply_color_coding_to_hsimgs(im_spec, lambdas)

im_spec_color_coded = zeros([size(im_spec,1), size(im_spec,2), 3, size(im_spec,3)]);

concat_size = [ceil(sqrt(numel(lambdas))), ceil(sqrt(numel(lambdas)))];
% concat_size = [6, 4];

[N,M,~] = size(im_spec);
% im_spec_color_coded_concat = zeros(size(im_spec,1)*concat_size, size(im_spec,2)*concat_size, 3 );
im_spec_color_coded_concat = zeros(size(im_spec,1)*concat_size(1), size(im_spec,2)*concat_size(2), 3 );

for i=1:numel(lambdas)
    wvls = lambdas(i);
    im_1channel = im_spec( :, :, i);
    
%     imwrite(im_1channel * 8, sprintf( '%s/%dnm.png', output_dir_gray, wvls));
    
    sRGB = spectrumRGB( wvls );
    im_spec_color_coded(:,:,1,i) = im_1channel .* sRGB(1);
    im_spec_color_coded(:,:,2,i) = im_1channel .* sRGB(2);
    im_spec_color_coded(:,:,3,i) = im_1channel .* sRGB(3);
    
    [c,r] = ind2sub([concat_size(2), concat_size(1)], i); % we swap the row and column
    min_r = (r-1)*N + 1; max_r = (r-1)*N + N;
    min_c = (c-1)*M + 1; max_c = (c-1)*M + M;
    im_spec_color_coded_concat(min_r:max_r, min_c:max_c,:) = im_spec_color_coded(:,:,:,i);
end

end