function [ imgrad ] = imgrad_any_dir( img, options, dir_map_xy )
%IMGRAD Computing an image gradient map
% Computing an image gradient map.
% RGB img (h,w,d) -> cat(3,dxR,dyR,dxG,dyG,dxB,dyB)
%
% [ imgrad ] = IMGRAD( img, boundary )
% Inputs
%   img: input image (h,w,d)
%   boundary: 'neumann' for Neumann boundary condition
%             'dirichlet' for Dirichlet boundary condition
% Outputs
%   imgrad: output gradient map (h,w,2*d)

[h,w,d] = size(img);
imgrad = zeros(h,w,d);


n_dir_map_xy = zeros(size(dir_map_xy));
n_dir_map_xy(:,:,1) = dir_map_xy(:,:,1) ./ sqrt(dir_map_xy(:,:,1).^2 + dir_map_xy(:,:,2).^2);
n_dir_map_xy(:,:,2) = dir_map_xy(:,:,2) ./ sqrt(dir_map_xy(:,:,1).^2 + dir_map_xy(:,:,2).^2);

[xx,yy] = meshgrid(1:w, 1:h);

xx2 = xx + n_dir_map_xy(:,:,1);
yy2 = yy + n_dir_map_xy(:,:,2);

xx3 = xx - n_dir_map_xy(:,:,1);
yy3 = yy - n_dir_map_xy(:,:,2);

switch options
    case 'forward'
        for i = 1:d
            nImg = interp2(xx, yy, img(:,:,i), xx2, yy2, 'linear', NaN);
            grad_i = nImg - img(:,:,i);
            grad_i(isnan(grad_i)) = 0;
            imgrad(:,:,i) = grad_i;
            % spline-based interpolation enables us to extrapolate the
            % samples outside of the region, while it requires more memory
            % and computation than cubic interpolation.
        end
    case 'backward'
        for i = 1:d
            nImg = interp2(xx, yy, img(:,:,i), xx3, yy3, 'linear', NaN);
            nImg(isnan(nImg)) = 0;
            grad_i = nImg - img(:,:,i);
            imgrad(:,:,i) = grad_i;

%             imgrad(:,:,i) = img(:,:,i) - interp2(xx, yy, img(:,:,i), xx2, yy2, 'spline');
        end
    otherwise
        error('Options must be forward or backward.');
end

end