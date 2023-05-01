function [result_final] = compute_ATA_mul_any_CTIS_vectorized(x_or_y, shifted_ind, spec2rgb, ref_ch_ind)
% spec2rgb: 3 * num_lambdas, spec2rgb converts the spectrum into rgb
% shifted_Ind: N * M * 2 * num_lambdas

% x: N * M * C, where N, M and C are row, col and # of channels
% projMat: N * M * C * 2
% mode 1: y = A * x

% BOUNDARY_FLAG = 1;

[N,M,~,nCh] = size(shifted_ind);
x_or_y = reshape(x_or_y, N, M, []);
% nCh = nCh + 1;
fprintf('.');




[xx,yy] = meshgrid(1:M, 1:N);
nCh_result = size(spec2rgb, 1);
x = x_or_y;

%% 1. Expand
result = zeros( 3*N,3*M,nCh );
%% 2. Disperse
for ch = 1:nCh
    
    %             xx2 = max( min(shifted_ind(:,:,1,ch), M), 1);
    %             yy2 = max( min(shifted_ind(:,:,2,ch), N), 1);
    %% 2. Disperse the masked image
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


y = result_final;
y = reshape( y, [], nCh_result);
ty = reshape((spec2rgb' * y')', 3*N, 3*M, nCh);
result_final = zeros( N, M, nCh );
for ch = 1:nCh
    up = imtranslate( squeeze(ty(1:N,(M+1):(2*M),ch)), (ch-ref_ch_ind)*[0, -1], 'FillValues', 0);
    down = imtranslate( squeeze(ty((2*N+1):(3*N),(M+1):(2*M),ch)), (ch-ref_ch_ind)*[0, 1], 'FillValues', 0);
    left = imtranslate( squeeze(ty((N+1):(2*N),1:M,ch)), (ch-ref_ch_ind)*[-1, 0], 'FillValues', 0);
    right = imtranslate( squeeze(ty((N+1):(2*N),(2*M+1):(3*M),ch)), (ch-ref_ch_ind)*[1, 0], 'FillValues', 0);
    center = ty((N+1):(2*N),(M+1):(2*M),ch);
    result_final(:,:,ch) = up + down + left + right + center;
end

result_final = result_final(:);
end