function [result_final] = compute_A_mul_any_CTIS(x_or_y, shifted_ind, spec2rgb, mode, ref_ch_ind, is_xy)
% spec2rgb: 3 * num_lambdas, spec2rgb converts the spectrum into rgb
% shifted_Ind: N * M * 2 * num_lambdas

% x: N * M * C, where N, M and C are row, col and # of channels
% projMat: N * M * C * 2
% mode 1: y = A * x
% mode 2: x = transpose(A) * y
if nargin < 6
    is_xy = 0;
end



BOUNDARY_FLAG = 1;

[N,M,~,nCh] = size(shifted_ind);
% nCh = nCh + 1;




[xx,yy] = meshgrid(1:M, 1:N);
nCh_result = size(spec2rgb, 1);

if mode == 1
    % shift the spectral images with depth-dependent displacement
    
    x = x_or_y;
%     x(1:N,1:M,:) = x_or_y;
%     x = x .* coded_mask;
    if is_xy == 0
        %% 1. Expand
        result = zeros( 3*N,3*M,nCh );
        %% 2. Disperse
        for ch = 1:nCh
            
            %             xx2 = max( min(shifted_ind(:,:,1,ch), M), 1);
            %             yy2 = max( min(shifted_ind(:,:,2,ch), N), 1);
            %% 2. Disperse the masked image
%             xx2 = shifted_ind(:,:,1,ch);
%             yy2 = shifted_ind(:,:,2,ch);
            % four directions
%             up = rot90(interp2( xx, yy, rot90(x(:,:,ch),-1), xx2, yy2, 'linear', 0),1);
%             down = rot90(interp2( xx, yy, rot90(x(:,:,ch),1), xx2, yy2, 'linear', 0),-1);
%             left = rot90(interp2( xx, yy, rot90(x(:,:,ch),-2), xx2, yy2, 'linear', 0),2);
%             right = interp2( xx, yy, x(:,:,ch), xx2, yy2, 'linear', 0);
            
            up = imtranslate( squeeze(x(:,:,ch)), (ch-ref_ch_ind)*[0, 1], 'FillValues', 0);
            down = imtranslate( squeeze(x(:,:,ch)), (ch-ref_ch_ind)*[0, -1], 'FillValues', 0);
            left = imtranslate( squeeze(x(:,:,ch)), (ch-ref_ch_ind)*[1, 0], 'FillValues', 0);
            right = imtranslate( squeeze(x(:,:,ch)), (ch-ref_ch_ind)*[-1, 0], 'FillValues', 0);
            
            %% 3. Copy and translate
            result(1:N,(M+1):(2*M),ch) = up;
            result((2*N+1):(3*N),(M+1):(2*M),ch) = down;
            result((N+1):(2*N),1:M,ch) = left;
            result((N+1):(2*N),(2*M+1):(3*M),ch) = right;
            result((N+1):(2*N),(M+1):(2*M),ch) = x(:,:,ch);
        end
        %% 4. Projection
        result = reshape(result, [], nCh);
        result_final = reshape( (spec2rgb * result')', 3*N, 3*M, nCh_result);
    else
        result = zeros(N, M, 2*nCh );
        for ch = 1:nCh
            %             xx2 = max( min(shifted_ind(:,:,1,ch), M), 1);
            %             yy2 = max( min(shifted_ind(:,:,2,ch), N), 1);
            xx2 = shifted_ind(:,:,1,ch);
            yy2 = shifted_ind(:,:,2,ch);
            
            newImg1 = interp2( xx, yy, x(:,:,2*ch-1), xx2, yy2, 'linear', 0);
            newImg2 = interp2( xx, yy, x(:,:,2*ch), xx2, yy2, 'linear', 0);
            
            result(:,:,2*ch-1) = newImg1;
            result(:,:,2*ch) = newImg2;
        end
        result = reshape(result, N*M, 2*nCh);
        result_final = zeros(N, M, nCh_result*2);
        result_final(:,:, 1:2:end) = reshape( (spec2rgb * result(:,1:2:end)')', N, M, nCh_result);
        result_final(:,:, 2:2:end) = reshape( (spec2rgb * result(:,2:2:end)')', N, M, nCh_result);
    end
%     result_final = result_final(1:params.size_original_h, 1:params.size_original_w, :);
elseif mode == 2
    %% (spec2rgb^T) y
    % increase the dimension of the RGB image into the dimension of the spectral images
    y = x_or_y;
    if is_xy == 0
        y = reshape( y, [], nCh_result);
        ty = reshape((spec2rgb' * y')', 3*N, 3*M, nCh);
        result_final = zeros( N, M, nCh );
        for ch = 1:nCh
            %             xx2 = max( min(xx + (xx - shifted_ind(:,:,1,ch)), M), 1);
            %             yy2 = max( min(yy + (yy - shifted_ind(:,:,2,ch)), N), 1);
                        xx2 = xx + (xx - shifted_ind(:,:,1,ch));
                        yy2 = yy + (yy - shifted_ind(:,:,2,ch));
%             xx2 = shifted_ind(:,:,1,ch);
%             yy2 = shifted_ind(:,:,2,ch);
            
            
%             up = rot90(interp2( xx, yy, rot90(ty(1:N,(M+1):(2*M),ch),-1), xx2, yy2, 'linear', 0),1);
%             down = rot90(interp2( xx, yy, rot90(ty((2*N+1):(3*N),(M+1):(2*M),ch),1), xx2, yy2, 'linear', 0),-1);
%             left = rot90(interp2( xx, yy, rot90(ty((N+1):(2*N),1:M,ch),-2), xx2, yy2, 'linear', 0),2);
%             right = interp2( xx, yy, ty((N+1):(2*N),(2*M+1):(3*M),ch), xx2, yy2, 'linear', 0);
            
            up = imtranslate( squeeze(ty(1:N,(M+1):(2*M),ch)), (ch-ref_ch_ind)*[0, -1], 'FillValues', 0);
            down = imtranslate( squeeze(ty((2*N+1):(3*N),(M+1):(2*M),ch)), (ch-ref_ch_ind)*[0, 1], 'FillValues', 0);
            left = imtranslate( squeeze(ty((N+1):(2*N),1:M,ch)), (ch-ref_ch_ind)*[-1, 0], 'FillValues', 0);
            right = imtranslate( squeeze(ty((N+1):(2*N),(2*M+1):(3*M),ch)), (ch-ref_ch_ind)*[1, 0], 'FillValues', 0);
            
            center = ty((N+1):(2*N),(M+1):(2*M),ch);
            result_final(:,:,ch) = up + down + left + right + center;
        end
    else
        y = reshape( y, N*M, nCh_result*2);
        ty1 = reshape((spec2rgb' * y(:,1:2:end)')', N, M, nCh);
        ty2 = reshape((spec2rgb' * y(:,2:2:end)')', N, M, nCh);
        result_final = zeros( N, M, 2*nCh );
        for ch = 1:nCh
            %             xx2 = max( min(xx + (xx - shifted_ind(:,:,1,ch)), M), 1);
            %             yy2 = max( min(yy + (yy - shifted_ind(:,:,2,ch)), N), 1);
                        xx2 = xx + (xx - shifted_ind(:,:,1,ch));
                        yy2 = yy + (yy - shifted_ind(:,:,2,ch));
%             xx2 = shifted_ind(:,:,1,ch);
%             yy2 = shifted_ind(:,:,2,ch);
            
            newImg1 = interp2( xx, yy, ty1(:,:,ch), xx2, yy2, 'linear', 0);
            newImg2 = interp2( xx, yy, ty2(:,:,ch), xx2, yy2, 'linear', 0);
            result_final(:,:,2*ch-1) = newImg1;
            result_final(:,:,2*ch) = newImg2;
        end
    end
%     result_final = result_final .* coded_mask;
else
    fprintf('mode should be one of {1, 2, 3}\n');
end

end