function [ imgrad ] = imgrad( img, boundary, xy )
%IMGRAD Computing an image gradient map
% Computing an image gradient map.
% RGB img (h,w,d) -> cat(3,dxR,dyR,dxG,dyG,dxB,dyB)
% 
% [ imgrad ] = IMGRAD( img, boundary )
% Inputs
%   img: input image (h,w,d)
%   boundary: 'neumann' for Neumann boundary condition
%             'dirichlet' for Dirichlet boundary condition
%   xy: 1 => Dx: horizontal
%       2 => Dy: vertical
%       3 => Dxy: horizontal, vertical
% Outputs
%   imgrad: output gradient map (h,w,2*d)

if nargin < 2
    boundary = 'neumann';
    xy = 3;
end
if nargin < 3
    xy = 3;
end

[h,w,d] = size(img);
if xy == 3
    imgrad = zeros(h,w,2*d);
else
    imgrad = zeros(h,w,d);
end

switch boundary
    case 'neumann'
        for i = 1:d
            if xy == 3
                imgrad(:,:,(2*i-1):(2*i)) = cat(3,[diff(img(:,:,i),1,2) zeros(h,1)],[diff(img(:,:,i));zeros(1,w)]);
            elseif xy == 1
                imgrad(:,:,i) = cat(3,[diff(img(:,:,i),1,2) zeros(h,1)]);
            elseif xy == 2
                imgrad(:,:,i) = cat(3,[diff(img(:,:,i));zeros(1,w)]);
            end
        end
    case 'dirichlet'
    otherwise
end

end