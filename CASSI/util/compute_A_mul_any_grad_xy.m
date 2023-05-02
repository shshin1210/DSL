function [result] = compute_A_mul_any_grad_xy(x_or_y, shifted_ind, spec2rgb_xy, mode)
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
    result = zeros( N, M, nCh * 2 );
    for ch = 1:nCh
        
        newImg_x = interp2( xx, yy, x(:,:,ch*2 - 1), shifted_ind(:,:,1,ch), shifted_ind(:,:,2,ch), 'linear', 0);
        newImg_y = interp2( xx, yy, x(:,:,ch*2 ), shifted_ind(:,:,1,ch), shifted_ind(:,:,2,ch), 'linear', 0);

        
        result(:,:,ch*2 - 1) = newImg_x;
        result(:,:,ch*2 ) = newImg_y;
    end
    
    % convert the spectral images into a RGB image
    result = reshape(result, N*M, 2 * nCh);
    result = reshape( (spec2rgb_xy * result')', N, M, 6);
    
elseif mode == 2
    %% (spec2rgb^T) y
    % increase the dimension of the RGB image into the dimension of the spectral images
    y = x_or_y;
    y = reshape( y, N*M, 3 * 2);
    ty = reshape((spec2rgb_xy' * y')', N, M, nCh * 2);
    result = zeros( N, M, nCh * 2 );
    
    % inversely shift the image
    for ch = 1:nCh
        newImg_x = interp2( xx, yy, ty(:,:,ch*2 - 1), xx + (xx - shifted_ind(:,:,1,ch)), yy + (yy - shifted_ind(:,:,2,ch)), 'linear', 0);
        newImg_y = interp2( xx, yy, ty(:,:,ch*2 ), xx + (xx - shifted_ind(:,:,1,ch)), yy + (yy - shifted_ind(:,:,2,ch)), 'linear', 0);

        result(:,:,ch*2 - 1) = newImg_x;
        result(:,:,ch*2 ) = newImg_y;
    end
else
    fprintf('mode should be one of {1, 2, 3}\n');
end

end