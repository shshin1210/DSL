function [ out ] = spectral_gradient( in, options, step_size, mode )
%SPECTRAL_GRADIENT Gradient operator in a spectral domain
%   Gradient operator in a spectral domain with Neumann boundary condition.
%    Difference between adjacent spectra is computed.
%
%    [ out ] = SPECTRAL_GRADIENT( in, options )
%    Inputs
%      in: input spectral image (h,w,lambda)
%      options
%        'forward': spectral gradient (A) (forward difference)
%           out = [in(2)-in(1);in(3)-in(2);...;in(n)-in(n-1);zeros(len)];
%        'backward': spectral divergence (AT) (backward difference)
%           out = [-in(1);in(1)-in(2);in(2)-in(3);...;in(n-1)-in(n)];
%      mode
%         1: size of in is equal to original hyperspectral data
%         2: size of in is doubled due to the xy gradient operator
%    Outputs
%      out: output (h,w,lambda)
if ~exist('mode', 'var')
    mode = 1;
end


if ndims(in) ~= 3
    if ndims(in) == 2
        in = reshape(in, size(in,1), 1, size(in,2));
    else
        error('input dimension of spectral gradient should be two or three.');
    end
end

[h,w,d] = size(in);
if mode == 2
    d = d/2;
end




%step_size = 1;
% step_size = 1;

switch options
    case 'forward'
        if mode == 1
            diff_out = in(:,:,(1+step_size):end) - in(:,:,1:(end-step_size));
            out = cat(3, diff_out, zeros(h,w,step_size));
        elseif mode == 2
            in2 = in(:,:,1:2:end);
            in3 = in(:,:,2:2:end);
            diff_out2 = in2(:,:,(1+step_size):end) - in2(:,:,1:(end-step_size));
            diff_out3 = in3(:,:,(1+step_size):end) - in3(:,:,1:(end-step_size));
            out = [];
            for i = 1:(d-1)
                out = cat(3, out, cat(3, diff_out2(:,:,i), diff_out3(:,:,i)) );
            end
            out = cat(3, out, zeros(h,w,2*step_size));
        end
    case 'backward'
        if mode == 1
            diff_out = in(:,:,1:(end-step_size)) - in(:,:,(1+step_size):end);
            out = cat(3, -in(:,:,1:step_size), diff_out);
        elseif mode == 2
            in2 = in(:,:,1:2:end);
            in3 = in(:,:,2:2:end);
            diff_out2 = in2(:,:,1:(end-step_size)) - in2(:,:,(1+step_size):end);
            diff_out3 = in3(:,:,1:(end-step_size)) - in3(:,:,(1+step_size):end);
            out = [];
            for i = 1:(d-1)
                out = cat(3, out, cat(3, diff_out2(:,:,i), diff_out3(:,:,i)) );
            end
            out = cat(3, cat(3, -in2(:,:,1:step_size), -in3(:,:,1:step_size)), out);            
        end
        %         out = in - circshift(in, step_size, 3);
        
    otherwise
        error('Options must be forward or backward.');
end

end