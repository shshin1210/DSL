function [result] = compute_A_mul_any_shiftonly(x_or_y, shifted_ind, mode)
% spec2rgb: 3 * num_lambdas, spec2rgb converts the spectrum into  RGB
% shifted_Ind: N * M * 2 * num_lambdas

% x: N * M * C, where N, M and C are row, col and # of channels
% projMat: N * M * C * 2
% mode 1: y = A * x
% mode 2: x = transpose(A) * y

[N,M,~,nCh] = size(shifted_ind);
% nCh = nCh + 1;

[xx,yy] = meshgrid(1:M, 1:N);

if mode == 1
    % shift the spectral images with depth-dependent displacement
    x = x_or_y;
    result = zeros( N, M, nCh );
    for ch = 1:nCh
%         newImg_spline = interp2( xx, yy, x(:,:,ch), shifted_ind(:,:,1,ch), shifted_ind(:,:,2,ch), 'spline');
        newImg_linear = interp2( xx, yy, x(:,:,ch), shifted_ind(:,:,1,ch), shifted_ind(:,:,2,ch), 'linear', 0);
        
%         newImg = zeros(size(newImg_linear));
%         newImg(newImg_linear==-1) = newImg_spline(newImg_linear==-1);
%         newImg(newImg_linear~=-1) = newImg_linear(newImg_linear~=-1);

        result(:,:,ch) = newImg_linear;
    end
    
    % convert the spectral images into a RGB image
    result = reshape(result, N, M, nCh);
    
elseif mode == 2
    %% (spec2rgb^T) y
    % increase the dimension of the RGB image into the dimension of the spectral images
    y = x_or_y;
    ty = reshape(y, N, M, nCh);
    result = zeros( N, M, nCh );
    
    % inversely shift the image
    for ch = 1:nCh
        %         newImg_spline = interp2( xx, yy, ty(:,:,ch), xx + (xx - shifted_ind(:,:,1,ch)), yy + (yy - shifted_ind(:,:,2,ch)), 'spline');
        %         newImg_linear = interp2( xx, yy, ty(:,:,ch), xx + (xx - shifted_ind(:,:,1,ch)), yy + (yy - shifted_ind(:,:,2,ch)), 'linear', 0);
%         newImg_linear = interp2( xx, yy, ty(:,:,ch), xx + (xx - shifted_ind(:,:,1,ch)), yy + (yy - shifted_ind(:,:,2,ch)), 'linear', 0);
        newImg_linear = interp2( xx, yy, ty(:,:,ch), shifted_ind(:,:,1,ch), shifted_ind(:,:,2,ch), 'linear', 0);

%         newImg = zeros(size(newImg_linear));
%         newImg(newImg_linear==-1) = newImg_spline(newImg_linear==-1);
%         newImg(newImg_linear~=-1) = newImg_linear(newImg_linear~=-1);

        result(:,:,ch) = newImg_linear;
    end
else
    fprintf('mode should be one of {1, 2, 3}\n');
end

end