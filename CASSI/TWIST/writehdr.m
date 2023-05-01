%============================================================================
%
% Codename: Bird Scanner (Yale Computer Graphics Group)
%
% Copyright (C) 2011-12 Min H. Kim. All rights reserved.
%
% GNU General Public License Usage
% Alternatively, this file may be used under the terms of the GNU General
% Public License version 3.0 as published by the Free Software Foundation
% and appearing in the file LICENSE.GPL included in the packaging of this
% file. Please review the following information to ensure the GNU General
% Public License version 3.0 requirements will be met:
% http://www.gnu.org/copyleft/gpl.html.
%
%============================================================================
% Write_RGBe.m
% save the data as a rgbe
% ver.2
% it is based on Ward, G. C++ source.
% and some codes is adopted from Walters,B., Taplin, L.A.
%============================================================================
% function writehdr(img, filename)
function writehdr(img, filename)
    % file save
    if nargin==1;
        [filename, filepath] = uiputfile('*.hdr', 'Save the HDRi as');
        if filename==0; return; end;
        filename = strcat(filepath,filename);
    end
    
    exposure = 0;
    % Extract R, G, B components
    dims=size(img);
    height=dims(1);
    width=dims(2);
    % Caution! it must be tranposed!!! 
    img = transpose3(img);
    float = reshape(img,height*width,3);
    rgbe = float2rgbe(float);
    
    % Interleave the channels to a linear output array
    hdr=uint8(zeros([1,width*height*4]));
    hdr(1:4:end) = rgbe(:,1);
    hdr(2:4:end) = rgbe(:,2);
    hdr(3:4:end) = rgbe(:,3);
    hdr(4:4:end) = rgbe(:,4);

    % Write the output file
    fid=fopen(filename,'w');
    % don't remove the file format definition! 
    fprintf(fid,'#?KIM05(RGB)\nSOFTWARE=Matlab-generated HDR file\n');
    if (nargin==2)
        fprintf(fid,'EXPOSURE=%d\n',exposure);
    end
    fprintf(fid,'FORMAT=32-bit_rle_rgbe\n');
    fprintf(fid,'\n-Y %d +X %d\n',height,width); 
    fwrite(fid,hdr,'uint8');
    fclose(fid);
end

function [rgbe] = float2rgbe(rgb) 
    s = size(rgb);
    rgb = reshape(rgb,prod(s)/3,3);
    rgbe = reshape(repmat(uint8(0),[s(1:end-1),4]),prod(s)/3,4);
    v = max(rgb,[],2); %find max rgb
    l = find(v>1e-32); %find non zero pixel list
    % it shoud be log(v(l)) in order to cover log Zero error
    rgbe(l,4) = uint8(round(128.5+log(v(l))/log(2))); %find E
    rgbe(l,1:3) = uint8(rgb(l,1:3)./repmat(2.^(double(rgbe(l,4))-128-8),1,3)); %find rgb multiplier
    reshape(rgbe,[s(1:end-1),4]); %reshape back to original dimensions
end
    
function [rgb] = rgbe2float(rgbe)
    s = size(rgbe);
    rgbe = reshape(rgbe,prod(s)/4,4);
    rgb = zeros(prod(s)/4,3);
    l = find(rgbe(:,4)>0); %nonzero pixel list
    rgb(l,:) = double(rgbe(l,1:3)).*repmat(2.^(double(rgbe(l,4))-128-8),1,3);
    rgb = reshape(rgb,[s(1:end-1),3]);
end
    
function mat2=transpose3(mat)
    m1=mat(:,:,1);
    m2=mat(:,:,2);
    m3=mat(:,:,3);
    
    m1a=transpose(m1);
    m2a=transpose(m2);
    m3a=transpose(m3);
    
    mat2(:,:,1)=m1a;
    mat2(:,:,2)=m2a;
    mat2(:,:,3)=m3a;
end