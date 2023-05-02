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
% function cassiwriteimages(refwhite, type, nopad, islinear, sourcepath)
function cassiwriteimages(refwhite, type, nopad, islinear, sourcepath)
    pathstr = '.';
    if nargin < 1
        refwhite = 0;
    end
    if nargin < 2
        type = 'color';
    end
    if nargin < 3
        nopad = 0;
    end
    if nargin <4
        islinear = 0;
    end
    % ask filename for the reconstructed listfile
    if nargin<5
        % file read
        [file_name,file_path] = uigetfile({'*.mat','mat';},...
            'Choose reconstructed file', 'MultiSelect', 'on');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
        
        if iscell(file_name)
            for i=1:size(file_name,2)
                sourcepath = char(strcat(file_path,file_name{i}));   
            end
        else
            sourcepath = strcat(file_path,file_name);   
        end
    end

    if size(refwhite,2)==2
        temp = refwhite(:,2);
        refwhite = temp;
    end
    
    disp(['>> ' sourcepath]);
    disp 'Reading data...';
    load(sourcepath);
    disp 'Done.';

    if strcmp(type,'gray') || strcmp(type,'cindex')
        % white balancing
        wx_twist = R2WBR(x_twist,refwhite);        
        % for gray and color
        for i=1:size(wx_twist,3)
            img(:,:) = wx_twist(:,:,i);
            img(img<0) = 0;
            img(img>1) = 1;

            [pathstr, name, ext] = fileparts(sourcepath);
            % if color
            if strcmp(type,'cindex')
                if (wvls2b(i)<400)
                    sRGB=spectrumRGB(400);
                elseif (wvls2b(i)>700)
                    sRGB=spectrumRGB(700);
                else
                    sRGB=spectrumRGB(wvls2b(i));
                end
                % normalize scalar
                sRGB = sRGB./max(sRGB); % --> this scalar doesn't matter (this color is just an index)

                color(:,:,1) = img.*sRGB(1);
                color(:,:,2) = img.*sRGB(2);
                color(:,:,3) = img.*sRGB(3);            
                name = regexprep(name, 'final_', ['cindex_' num2str(floor(wvls2b(i))) 'nm_'],'ignorecase');
            else % for gray
                color(:,:,1) = img;
                color(:,:,2) = img;
                color(:,:,3) = img; 
                name = regexprep(name, 'final_', ['final_each_' num2str(floor(wvls2b(i))) 'nm_'],'ignorecase');
            end
            if isempty(pathstr)
                pathstr = '.';
            end
            fn = [pathstr, '/', name, '.png'];      
            color = color.^(1/2.2);


            %==================================================================
            % zero padding
            % x_twist --> final results
            % throwing the results into the black data.
            if (nopad == 1)
                zeropad = color;
            else
                zeropad = zeros([totalres totalres size(color,3)], 'double');
                zeropad(rect(2):rect(2)+rect(4) , rect(1):rect(1)+rect(3),:) = color;
            end
            %==================================================================        
            imwrite(zeropad, fn, 'png');
            disp(['writing ' fn]);
        end
    elseif strcmp(type,'tripack')
        % white balancing
        wx_twist = R2WBR(x_twist,refwhite);        
        % packing three colors
        for i=1:3:size(wx_twist,3)
            for j=1:3
                % first channel
                if (i+j-1>size(wx_twist,3))
                    imgt = zeros([size(wx_twist,1),size(wx_twist,2)]);
                    names(j) = 0;
                else
                    imgt(:,:) = wx_twist(:,:,i+j-1);
                    imgt(imgt<0) = 0;
                    imgt(imgt>1) = 1;
                    names(j) = floor(wvls2b(i+j-1));
                end
                img(:,:,j) = imgt; 
            end

            [pathstr, name, ext] = fileparts(sourcepath);
            color = img; 
            name = regexprep(name, 'final_', ['final_each_' num2str(names(1)) '_' num2str(names(2)) '_' num2str(names(3)) 'nm_'],'ignorecase');
            if isempty(pathstr)
                pathstr = '.';
            end
            fn = [pathstr, '/', name, '.png'];      
            color = color.^(1/2.2);

            %==================================================================
            % zero padding
            % x_twist --> final results
            % throwing the results into the black data.
            if (nopad == 1)
                zeropad = color;
            else
                zeropad = zeros([totalres totalres size(color,3)], 'double');
                zeropad(rect(2):rect(2)+rect(4) , rect(1):rect(1)+rect(3),:) = color;
            end
            %==================================================================        
            imwrite(zeropad, fn, 'png');
            disp(['writing ' fn]);
        end
    elseif strcmp(type,'color')
        disp('converting spectra to sRGB...');
        sRGB = R2sRGB(wvls2b,x_twist,refwhite, islinear);

        [pathstr, name, ext] = fileparts(sourcepath);
        name = regexprep(name, 'final_', ['final_color_' num2str(floor(wvls2b(1))) '-'  num2str(floor(wvls2b(size(wvls2b,2)))) 'nm_'],'ignorecase');
         if isempty(pathstr)
            pathstr = '.';
         end
        fn = [pathstr, '/', name, '.png'];        

        %==================================================================
        % zero padding
        % x_twist --> final results
        % throwing the results into the black data.
        if (nopad == 1)
            zeropad = sRGB;
        else
            zeropad = zeros([totalres totalres size(sRGB,3)], 'double');
            zeropad(rect(2):rect(2)+rect(4) , rect(1):rect(1)+rect(3),:) = sRGB;
        end
        %==================================================================
        imwrite(zeropad, fn, 'png');  
        % show the final result
        imshow(zeropad);
    elseif strcmp(type,'hdr')
        disp('converting spectra to sRGB HDR...');
        sRGB = R2sRGB(wvls2b,x_twist,refwhite, islinear);

        [pathstr, name, ext] = fileparts(sourcepath);
        name = regexprep(name, 'final_', ['final_color_' num2str(floor(wvls2b(1))) '-'  num2str(floor(wvls2b(size(wvls2b,2)))) 'nm_'],'ignorecase');
         if isempty(pathstr)
            pathstr = '.';
         end
        fn = [pathstr, '/', name, '.hdr'];        

        %==================================================================
        % zero padding
        % x_twist --> final results
        % throwing the results into the black data.
        if (nopad == 1)
            zeropad = sRGB;
        else
            zeropad = zeros([totalres totalres size(sRGB,3)], 'double');
            zeropad(rect(2):rect(2)+rect(4) , rect(1):rect(1)+rect(3),:) = sRGB;
        end
        %==================================================================
        writehdr(zeropad, fn);  
        % show the final result
        imshow(zeropad);
    elseif strcmp(type,'graysum')
        % white balancing
        wx_twist = R2WBR(x_twist,refwhite); 
        
        % sum of energy
        sumimg = sum(wx_twist,3);

        % Gaussian blur (doesn't make big difference)
        hsize = [3 3];
        sigma = 0.5;
        %h = fspecial('gaussian', hsize, sigma);
        h = fspecial('disk',10);
        blurred = imfilter(sumimg,h,'replicate');

        minsum = min(min(blurred));
        maxsum = max(max(blurred));

        %normimg = (sumimg - minsum)/(maxsum - minsum);
        normimg = (sumimg)/(maxsum);
        normimg(normimg>1) = 1.0;
        normimg(normimg<0) = 0.0;
        normimg = normimg.^(1/2.2);

        [pathstr, name, ext] = fileparts(sourcepath);
        name = regexprep(name, 'final_', ['final_sum_' num2str(floor(wvls2b(1))) '-'  num2str(floor(wvls2b(size(wvls2b,2)))) 'nm_'],'ignorecase');
        if isempty(pathstr)
            pathstr = '.';
        end
        fn = [pathstr, '/', name, '.png'];        
        %==================================================================
        % zero padding
        % x_twist --> final results
        % throwing the results into the black data.
        if (nopad == 1)
            zeropad = normimg;
        else
            zeropad = zeros([totalres totalres size(normimg,3)], 'double');
            zeropad(rect(2):rect(2)+rect(4) , rect(1):rect(1)+rect(3),:) = normimg;
        end
        %==================================================================
        imwrite(zeropad, fn, 'png'); 
    else
        return;
    end
end
