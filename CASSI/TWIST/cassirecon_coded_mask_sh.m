function [im_spec, obj_twist] = cassirecon_coded_mask_sh(im_observed, shifted_ind, spec2grey, coded_mask, params)
%% Prepare Parameters

tau = params.tau; %0.15; % smoothing factor parameters
tv_iter = params.tv_iter; % numger of iteration in a single denoise (for tvdenoise.m)
iterA = params.iterA; % max iteration for TwIST
tolA = params.tolA; % Iteration stop criteria in TwIST


y = im_observed; % observed image: in our case RGB image
% [n1, n2, ~] = size(im_observed_rgb); % row, column

QPhi = @(x)compute_A_mul_any_coded_mask(x, shifted_ind, spec2grey, 1, coded_mask);
QPhi_T= @(y)compute_A_mul_any_coded_mask(y, shifted_ind, spec2grey, 2, coded_mask);
Psi = @(x,th) cassidenoise(x,th,tv_iter);
Phi = @(x) TVnorm3D(x);

% Reconstruct using TwIST
[x_recon,dummy,obj_twist,times_twist,dummy,mse_twist] = TwIST( ...
    y,QPhi,tau,...
    'AT', QPhi_T, ...
    'Psi', Psi, ...
    'Phi',Phi, ...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',iterA,...
    'ToleranceA',tolA,...
    'Debias',0,...
    'Verbose', 1,...
    'lambda', 1e-6);

im_spec = x_recon;

end



function y = TVnorm3D(x)
% this is phi function (this produces the summation of the magnitudes of gradients)
% TVnonmspectralimging --> one constant
m=size(x,3);

shift1 = circshift(x, [-1 0 0]);
shift2 = circshift(x, [0 -1 0]);

sub_x = x(1:end-1,1:end-1,:);
shift1 = shift1(1:end-1,1:end-1,:);
shift2 = shift2(1:end-1,1:end-1,:);

% L-2 norm [SJ]
% diff1 = (sub_x - shift1) .^ 2;
% diff2 = (sub_x - shift2) .^ 2;
% y = diff1 + diff2;
% y = sqrt(y);
% y = sum(y(:));

% L-1 norm [MK] --> slightly better!
diff1 = abs(sub_x - shift1);
diff2 = abs(sub_x - shift2);

y = diff1 + diff2;
y = sum(y(:));

end




% 
% function [result] = compute_A_mul_any_v3(x_or_y, shifted_ind, spec2rgb)
% % spec2rgb: 3 * num_lambdas, spec2rgb converts the spectrum into  RGB
% % shifted_Ind: N * M * 2 * num_lambdas
% 
% % x: N * M * C, where N, M and C are row, col and # of channels
% % projMat: N * M * C * 2
% % mode 1: y = A * x
% % mode 2: x = transpose(A) * y
% 
% [N,M,~,nCh] = size(shifted_ind);
% % nCh = nCh + 1;
% 
% [xx,yy] = meshgrid(1:M, 1:N);
% 
% % shift the spectral images with depth-dependent displacement
% x = x_or_y;
% result = zeros( N, M, nCh );
% for ch = 1:nCh
%     %         if ch == 1
%     %             newImg  = x(:,:,1);
%     %         else
%     %             newImg = interp2( xx, yy, x(:,:,ch), shifted_ind(:,:,2,ch-1), shifted_ind(:,:,1,ch-1), 'linear', 0);
%     %         end
%     newImg = interp2( xx, yy, x(:,:,ch), shifted_ind(:,:,1,ch), shifted_ind(:,:,2,ch), 'linear', 0);
%     result(:,:,ch) = newImg;
% end
% 
% % convert the spectral images into a RGB image
% result = reshape(result, N*M, nCh);
% result = reshape( (spec2rgb * result')', N, M, 3);
% 
% end
% 
% function [result] = compute_AT_mul_any_v3(x_or_y, shifted_ind, spec2rgb)
% % (spec2rgb^T) y
% % increase the dimension of the RGB image into the dimension of the spectral images
% [N,M,~,nCh] = size(shifted_ind);
% [xx,yy] = meshgrid(1:M, 1:N);
% 
% y = x_or_y;
% y = reshape( y, N*M, 3);
% ty = reshape((spec2rgb' * y')', N, M, nCh);
% result = zeros( N, M, nCh );
% 
% % inversely shift the image
% for ch = 1:nCh
%     %         if ch == 1
%     %             newImg = ty(:,:,1);
%     %         else
%     %             newImg = interp2( xx, yy, ty(:,:,ch), xx + (xx - shifted_ind(:,:,2,ch-1)), yy + (yy - shifted_ind(:,:,1,ch-1)), 'linear', 0);
%     %         end
%     newImg = interp2( xx, yy, ty(:,:,ch), xx + (xx - shifted_ind(:,:,1,ch)), yy + (yy - shifted_ind(:,:,2,ch)), 'linear', 0);
%     
%     result(:,:,ch) = newImg;
% end
% 
% end




%
% function y=R2(f,n1,n2,m,Cs,nt,disperses,flipped) % y = Ax (h*w*snap)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % [Min] Here multidimensional dot product was impremented
% %       as elementary product plus sum
% %       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% % Elementary product
% % Sum up the 3rd dimensions in each seperate image:
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% f=reshape(f,[n1,n2,m]);
% gp=repmat(f,[1 1 1 nt]).*Cs; % 4D * 4D
%
%
% % Disperse
% for i = 1:m
%     gp_1channels = gp( :, :, i, : );
%
%     for j = flipped
%         gp( :, :, i, j ) = circshift( gp_1channels(:,:,1,j), [ 0, -disperses(j, i), 0 ] );
%     end
%
%     other = 1:nt;
%     other(flipped) = [];
%     for j = other
%         gp( :, :, i, j ) = circshift( gp_1channels(:,:,1,j), [ 0, disperses(j, i), 0 ] );
%     end
% end
%
% % Projection
% y=sum(gp,3); % 4D -> 3D
% y=y(:); % vectorize
% end
%
% function f=RT2(y,n1,n2,m,Cs,nt,disperses,flipped) % f = ATy (h*w*spec)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % [Min] Here multidimensional dot product was impremented
% %       as elementary product plus sum
% %       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% % Elementary product
% % Sum up on the 4th dimension:
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% y=reshape(y,[n1,n2,1,nt]);
% yp=repmat(y,[1,1,m,1]);
%
%
% % Disperse
% for i = 1:int32( m )
%     yp_1channels = yp( :, :, i, : );
%
%     for j = flipped
%         yp( :, :, i, j ) = circshift( yp_1channels(:,:,1,j), [ 0, disperses(j, i), 0 ] );
%     end
%
%     other = 1:nt;
%     other(flipped) = [];
%     for j = other
%         yp( :, :, i, j ) = circshift( yp_1channels(:,:,1,j), [ 0, -disperses(j, i), 0 ] );
%     end
% end
%
% yp=yp.*Cs; % 4D * 4D
% f=sum(yp,4); % DUKE: 4D -> 3D
% end