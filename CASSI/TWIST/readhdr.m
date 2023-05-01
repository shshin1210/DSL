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
% HDRCAM09 HDR file reader
%============================================================================
% read_rgbe v.2.2
% it read the Run-Length Encoding RGBe format and without RLE encoding.
% it is based on Ward, G. C++ source and
% implementation of Walters,B., Taplin, L.A.
%============================================================================
% function [HDRi, fileinfo] = readhdr(filename)
function [HDRi, fileinfo] = readhdr(filename)
    % for compatibility with Linux
    %setappdata(0,'UseNativeSystemDialogs',0);
    if nargin==0;
        % file read
        [file_name,file_path] = uigetfile({'*.hdr','HDR';},'Choose an HDR image');
        %if iscell(file_name)==0;HDRi=0;fileinfo=0; return; end;
        %1D string array names of files 
        source_images = strcat(file_path,file_name);   
        filename = char(source_images);
    end
    %open the file
    fid = fopen(filename,'r');

    %read the magic number
    tline = fgetl(fid);
%    if length(tline)<3 | tline(1:2) ~= '#?'
%        error('invalid header');
%    end
    fileinfo.identifier = tline(3:end);

    %read the header variables into a structure
    tline = fgetl(fid);
    while ~isempty(tline)
        %find and set the variable name
        n = strfind(tline,'=');
        if ~isempty(n)
            vname = lower(tline(1:n(1)-1));
            vval = tline(n+1:end);
            fileinfo = setfield(fileinfo,vname,tline(n+1:end));
            % fprintf('Variable = %s, Value = %s\n',vname, vval);
        end
        %read the next line
        tline = fgetl(fid);
    end

    %set the resolution variables
    tline = fgetl(fid);
    fileinfo.Ysign = tline(1);
    [fileinfo.height,count,errmsg,nextindex] = sscanf(tline(4:end),'%d',1);
    fileinfo.Xsign = tline(nextindex+4);
    [fileinfo.width,count,errmsg,nextindex] = sscanf(tline(nextindex+7:end),'%d',1);
    % fprintf('resolution: %s\n',tline);

    %allocate space for the scan line data
    HDRi = zeros(fileinfo.height, fileinfo.width, 3);

    %read the remaining data
    [data, count] = fread(fid,inf,'uint8');
    fclose(fid);

    scanline_width = fileinfo.width;
    num_scanlines = fileinfo.height;

    if ((scanline_width < 8)|(scanline_width > 32767))
        % run length encoding is not allowed so read flat
        HDRi = rgbe2float(reshape(data,fileinfo.width,fileinfo.height,4));
        return;
    end

    scanline_buffer = repmat(uint8(0),scanline_width,4);
    dp = 1; %set the data pointer to the begining of the read data

    % read in each successive scanline */
    for scanline=1:num_scanlines
        if (data(dp) ~= 2) | (data(dp+1) ~= 2)% | (bitget(data(dp+2),8)~=1)
            % error('this file is not run length encoded');
            rgbe(:,1) = data(1:4:end);
            rgbe(:,2) = data(2:4:end);
            rgbe(:,3) = data(3:4:end);
            rgbe(:,4) = data(4:4:end);

            float = rgbe2float(rgbe);
            img = reshape(float, fileinfo.width, fileinfo.height, 3);
            HDRi = transpose3(img);
            return;
        end

        if (bitshift(data(dp+2),8)+data(dp+3))~=scanline_width % -128
            error('wrong scanline width');
        end

        dp = dp+4;

        % read each of the four channels read the scanline into the buffer
        for i=1:4 
            ptr = 1;
            while(ptr <= scanline_width)
                if (data(dp) > 128) % a run of the same value
                    count = data(dp)-128;
                    if ((count == 0)|(count-1 > scanline_width - ptr)) 
                        warning('bad scanline data');
                    end
                    scanline_buffer(ptr:ptr+count-1,i) = data(dp+1);
                    dp = dp+2;
                    ptr = ptr+count;
                else % a non-run
                    count = data(dp);
                    dp = dp+1;
                    if ((count == 0)|(count-1 > scanline_width - ptr)) 
                        warning('bad scanline data');
                    end
                    scanline_buffer(ptr:ptr+count-1,i) = data(dp:dp+count-1);
                    ptr = ptr+count;
                    dp = dp+count;
                end
            end
        end

        % now convert data from buffer into floats
        HDRi(scanline,:,:) = rgbe2float(scanline_buffer);
    end
    
% standard conversion from float pixels to rgbe pixels
% the last dimension is assumed to be color
function [rgbe] = float2rgbe(rgb) 
    s = size(rgb);
    rgb = reshape(rgb,prod(s)/3,3);
    rgbe = reshape(repmat(uint8(0),[s(1:end-1),4]),prod(s)/3,4);
    v = max(rgb,[],2); %find max rgb
    l = find(v>1e-32); %find non zero pixel list
    rgbe(l,4) = uint8(round(128.5+log(v)/log(2))); %find E
    rgbe(l,1:3) = uint8(rgb(l,1:3)./repmat(2.^(double(rgbe(l,4))-128-8),1,3)); %find rgb multiplier
    reshape(rgbe,[s(1:end-1),4]); %reshape back to original dimensions

% standard conversion from rgbe to float pixels
% note: Ward uses ldexp(col+0.5,exp-(128+8)). 
%       in the range [0,1] to map back into the range [0,1].
function [rgb] = rgbe2float(rgbe)
    s = size(rgbe);
    rgbe = reshape(rgbe,prod(s)/4,4);
    rgb = zeros(prod(s)/4,3);
    l = find(rgbe(:,4)>0); %nonzero pixel list
    rgb(l,:) = double(rgbe(l,1:3)).*repmat(2.^(double(rgbe(l,4))-128-8),1,3);
    rgb = reshape(rgb,[s(1:end-1),3]);
