function [result_final] = compute_A_mul_any_t_fixed(x_or_y, shifted_ind, spec2rgb, mode, is_xy)
% spec2rgb: 3 * num_lambdas, spec2rgb converts the spectrum into  RGB
% shifted_Ind: N * M * 2 * num_lambdas

% x: N * M * C, where N, M and C are row, col and # of channels
% projMat: N * M * C * 2
% mode 1: y = A * x
% mode 2: x = transpose(A) * y
if nargin < 5
    is_xy = 0;
end

BOUNDARY_FLAG = 1;

[N,M,~,nCh,~] = size(shifted_ind);
% nCh = nCh + 1;

[xx,yy] = meshgrid(1:M, 1:N);

if mode == 1
    % shift the spectral images with depth-dependent displacement
    x = x_or_y;
    if is_xy == 0
        result = zeros( N, M, nCh );
        for ch = 1:nCh
            
            %             xx2 = max( min(shifted_ind(:,:,1,ch), M), 1);
            %             yy2 = max( min(shifted_ind(:,:,2,ch), N), 1);
            xx2 = shifted_ind(:,:,1,ch, 1);
            yy2 = shifted_ind(:,:,2,ch, 1);
            %
            newImg = interp2( xx, yy, x(:,:,ch), xx2, yy2, 'linear', 0);
            result(:,:,ch) = newImg;
        end
        result = reshape(result, N*M, nCh);
        result_final = reshape( (spec2rgb * result')', N, M, 3);
    else
        result = zeros(N, M, 2*nCh );
        for ch = 1:nCh
            %             xx2 = max( min(shifted_ind(:,:,1,ch), M), 1);
            %             yy2 = max( min(shifted_ind(:,:,2,ch), N), 1);
            xx2 = shifted_ind(:,:,1,ch, 1);
            yy2 = shifted_ind(:,:,2,ch, 1);
            
            newImg1 = interp2( xx, yy, x(:,:,2*ch-1), xx2, yy2, 'linear', 0);
            newImg2 = interp2( xx, yy, x(:,:,2*ch), xx2, yy2, 'linear', 0);
            
            result(:,:,2*ch-1) = newImg1;
            result(:,:,2*ch) = newImg2;
        end
        result = reshape(result, N*M, 2*nCh);
        result_final = zeros(N, M, 6);
        result_final(:,:, 1:2:end) = reshape( (spec2rgb * result(:,1:2:end)')', N, M, 3);
        result_final(:,:, 2:2:end) = reshape( (spec2rgb * result(:,2:2:end)')', N, M, 3);
    end
    
elseif mode == 2
    %% (spec2rgb^T) y
    % increase the dimension of the RGB image into the dimension of the spectral images
    y = x_or_y;
    if is_xy == 0
        y = reshape( y, N*M, 3);
        ty = reshape((spec2rgb' * y')', N, M, nCh);
        result_final = zeros( N, M, nCh );
        for ch = 1:nCh
            %             xx2 = max( min(xx + (xx - shifted_ind(:,:,1,ch)), M), 1);
            %             yy2 = max( min(yy + (yy - shifted_ind(:,:,2,ch)), N), 1);
%             xx2 = xx + (xx - shifted_ind(:,:,1,ch, 2));
%             yy2 = yy + (yy - shifted_ind(:,:,2,ch, 2));
            xx2 = shifted_ind(:,:,1,ch, 2);
            yy2 = shifted_ind(:,:,2,ch, 2);

            newImg = interp2( xx, yy, ty(:,:,ch), xx2, yy2, 'linear', 0);
            result_final(:,:,ch) = newImg;
        end
    else
        y = reshape( y, N*M, 6);
        ty1 = reshape((spec2rgb' * y(:,1:2:end)')', N, M, nCh);
        ty2 = reshape((spec2rgb' * y(:,2:2:end)')', N, M, nCh);
        result_final = zeros( N, M, 2*nCh );
        for ch = 1:nCh
            %             xx2 = max( min(xx + (xx - shifted_ind(:,:,1,ch)), M), 1);
            %             yy2 = max( min(yy + (yy - shifted_ind(:,:,2,ch)), N), 1);
%             xx2 = xx + (xx - shifted_ind(:,:,1,ch));
%             yy2 = yy + (yy - shifted_ind(:,:,2,ch));
            xx2 = shifted_ind(:,:,1,ch, 2);
            yy2 = shifted_ind(:,:,2,ch, 2);

            newImg1 = interp2( xx, yy, ty1(:,:,ch), xx2, yy2, 'linear', 0);
            newImg2 = interp2( xx, yy, ty2(:,:,ch), xx2, yy2, 'linear', 0);
            result_final(:,:,2*ch-1) = newImg1;
            result_final(:,:,2*ch) = newImg2;
        end
    end
    
else
    fprintf('mode should be one of {1, 2, 3}\n');
end

end