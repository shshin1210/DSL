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
%function [refwhite,sRGB] = cassidisplay(refwhite, islinear, sourcepath)
function [refwhite,sRGB] = cassidisplay(refwhite, islinear, sourcepath)
    % ask filename for the reconstructed listfile
    if nargin<1
        refwhite = 0;
    end
    if nargin<2
        islinear=0;
    end
    
    % stretch colors by normalization with min and max
    isstretch = 1;
    
    if nargin<3
        % file read
        [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose a reconstructed file');
        % leave function when no file is selected.
        if size(file_path,2) == 1
            return;
        end
        sourcepath = char(strcat(file_path,file_name));   
    end

    if size(refwhite,2)==2
        temp = refwhite(:,2);
        refwhite = temp;
    end
    
    disp 'Reading data...';
    load(sourcepath);
    disp 'Done.';

    ind = 0;
    sRGB = 0;    
    sclr = 0.0001;

        
    %-----------------------------------------------------
    % this is the white scaling (normalization) part
    % Don't try to scale the values in other place
    %-----------------------------------------------------
    if size(refwhite,1)>2
        % input: raw radiance before white balancing
        sRGB = R2sRGB(wvls2b,x_recon,refwhite, islinear, isstretch);     
        % output: reflectance after calculation
        wx_recon = Rad2Ref(x_recon,refwhite);
        disp('Done.');
    elseif refwhite == 1 % reflectance mode
        disp 'Calculating reflectance now'
        %-------------------------------------------------------------------
        % white balancing
        % Select reference white
        h1 = msgbox('Select the reference white','Message','warn');beep;
        sRGB = R2sRGB(wvls2b,x_recon, 0, islinear, isstretch);
        
        h2 = figure; [y2 rect2]=imcrop(sRGB,[]);
        close(h2);

        if isempty(rect2)
            sclr = 0;
            return;
        end
        rect2=round(rect2);
        x_reconw=x_recon(rect2(2):rect2(2)+rect2(4) , rect2(1):rect2(1)+rect2(3) , :);
        % Average:
        x_reconw=mean(x_reconw,1); x_reconw=mean(x_reconw,2);
        x_reconw = x_reconw(:);
        % avoid singularity
        x_reconw(end) = x_reconw(end-1);
%         if x_reconw(end)<0.001 
%             x_reconw(end) = 0.001;
%         end
        
        % show plot
        disp(['------------------------------------------------------------']);
        disp(['(' num2str(ind) ') Reference White Measurement (wavelength[nm]; signals): ']);
        disp(['rect: ' num2str(rect2)]);

        % for real measurements
        for k=1:size(wvls2b,2)
            if x_reconw(k) < 0
                x_reconw(k) = 0;
            end
            disp([num2str(floor(wvls2b(k))) '; ' num2str(x_reconw(k))]);
        end    
        disp(['Aver. sum of energy: ' num2str(sum(x_reconw))]);
        %-----------------------------------------------------

        fh3 = figure(3);
        set(fh3,'Name','Radiance');
        clf(fh3); 

        set(gcf,'color','white');   
        hold on;
        plot(wvls2b,x_reconw,'.','MarkerEdgeColor','r');    % show real measurements
        plot(wvls2b,x_reconw, '--b');    % raw measurement

        % sending the data to output
        refwhite = x_reconw;
        whitefn = regexprep(sourcepath, '.mat', '_refWhite.mat','ignorecase');
        save(whitefn,'refwhite');

        xlabel('Wavelength[nm]');
        ylabel('Relative radiance');
        title('Spectral Power Distribution');

        % determine the maximum in the axes
        if (max(x_reconw)>sclr)
            sclr = max(x_reconw); % the most
        end
        ylim([0,sclr]);

        xlim([300,1010]);

        disp('Calculating reflectance from radiance...');
        
        sRGB = R2sRGB(wvls2b,x_recon,refwhite, islinear, isstretch);        
        % calculating reflectance (white balancing)
        wx_recon = Rad2Ref(x_recon,refwhite);

        disp('Done.');
        %-------------------------------------------------------------------
    else
        disp 'this is the radiance mode.'
        wx_recon = x_recon;
        sRGB = R2sRGB(wvls2b,x_recon,0,islinear, isstretch);
    end
    
    %-------------------------------------------------------------------
    %return;
    %================================================
    % Display Color Image
    %================================================
    output(:,1) = wvls2b;
    while 1
        [sclr,x_reconi] = toshow(wx_recon,wvls2b,sclr,sRGB,ind,refwhite);
        ind = ind + 1;        
        output(:,ind+1) = x_reconi;
        if (sclr == 0) 
            output = output(:,1:(end-1));
            save('measurement.mat', 'output');
            if checkfigs(3)
               close(3);
            end
            return;
        end
    end
end

function [sclr,x_reconi] = toshow(x_recon,wvls2b,sclr,sRGB,ind,refwhite)
    rect_post = 0;
    x_recon2=0;
    x_reconi=0;
    %================================================

    %================================================
    % Display one image data
    %================================================
    % figure
    % Multiple pts:
    fh1 = figure(1);
    set(fh1,'Name','Visualization');
    clf(fh1); 
    sumimg = sum(x_recon,3);
    
    [y2, rect2]=imcrop(sRGB);
    if isempty(rect2)
        sclr = 0;
        return;
    end
        
    rect2=round(rect2);
    x_recon2=x_recon(rect2(2):rect2(2)+rect2(4) , rect2(1):rect2(1)+rect2(3) , :);
    
    % Average:
    x_recon2=mean(x_recon2,1); x_recon2=mean(x_recon2,2);
    rect_post=rect2;

    x_recon2=squeeze(x_recon2);
    % eliminate the last noise
    x_recon2(end) = x_recon2(end-1);
    
    %-----------------------------------------------------
    % regularizating signal
%    cf = smoothsplinefitting(wvls2b,x_recon2);
    stwave = wvls2b(1);
    edwave = wvls2b(end);
%    count = (edwave-stwave)/10+1; %10nm interval
    count = (edwave-stwave)+1; % 1nm interval
    
    wvls2i = linspace(stwave,edwave,count);
    
%     disp(['------------------------------------------------------------']);
%     disp(['Measurement (wavelength[nm] - regularized relative radiance)']);
%     disp(['rect: ' num2str(rect2)]);
    % for regress line
%     for k=1:size(wvls2i,2)
%         x_recon2r(k) = cf(wvls2b(k));
%         temp = cf(wvls2i(k));
%         if temp<0
%             temp = 0;
%         end
%         x_recon2r(k) = temp;
% %         disp([num2str(floor(wvls2i(k))) '; ' num2str(x_recon2r(k))]);
%     end   
    
    disp(['------------------------------------------------------------']);
    disp(['(' num2str(ind) ') Measurement (wavelength[nm]; signals): ']);
    disp(['rect: ' num2str(rect2)]);
    
    % for real measurements
    for k=1:size(wvls2b,2)
        if x_recon2(k) < 0
            x_recon2(k) = 0;
        end
        %disp([num2str(floor(wvls2b(k))) '; ' num2str(x_recon2(k))]);
        %disp(num2str(x_recon2(k)));
    end  
    
    t = num2str(x_recon2');
    q = strsplit(t, '\s+', 'DelimiterType','RegularExpression');
    w = strjoin(q, '\n');
    clipboard('copy', w);
    
    disp(['Aver. sum of energy: ' num2str(sum(x_recon2))]);
    %-----------------------------------------------------
    fh3 = figure(3);
    set(fh3,'Name','Radiometric Measurement');
    clf(fh3); 
    
    set(gcf,'color','white');   
    hold on;
    plot(wvls2b,x_recon2,'.','MarkerEdgeColor','r');    % show real measurements
    %plot(wvls2i,x_recon2r, '--b');    % interpolation
    plot(wvls2b,x_recon2, '--b');    % raw measurement

    % sending the data to output
    x_reconi = x_recon2;
    
    xlabel('Wavelength[nm]');
    title('Spectral Power Distribution');
    
    % determine the maximum in the axes
    if (max(x_recon2)>sclr)
%        sclr = 0.1;
        sclr = max(x_recon2); % the most
    end
    
    
        %-----------------------------------------------------
    if (size(refwhite,1)>2) % reflectance mode
        ylim([0,1.2]);
        ylabel('Reflectance');
    else
        ylim([0,sclr]);
        ylabel('Radiance');
    end
    
    xlim([300,1010]);
end

function status = checkfigs(h)
% if any of the passed in handles are invalid or not figure handles, return false
status = true;
for i=1:length(h)
    if ~any(ishghandle(h(i),'figure'))
        status = false;
        return
    end
end
end
