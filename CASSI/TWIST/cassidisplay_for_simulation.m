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
function [refwhite,sRGB] = cassidisplay_for_simulation(refwhite, islinear, sourcepath)
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
        sRGB = R2sRGB(wvls2b,x_twist,refwhite, islinear, isstretch);     
        % output: reflectance after calculation
        wx_twist = Rad2Ref(x_twist,refwhite);
        disp('Done.');
    elseif refwhite == 1 % reflectance mode
        disp 'Calculating reflectance now'
        %-------------------------------------------------------------------
        % white balancing
        % Select reference white
        h1 = msgbox('Select the reference white','Message','warn');beep;
        sRGB = R2sRGB(wvls2b,x_twist, 0, islinear, isstretch);
        
        h2 = figure; [y2 rect2]=imcrop(sRGB,[]);
        close(h2);

        if isempty(rect2)
            sclr = 0;
            return;
        end
        rect2=round(rect2);
        x_twistw=x_twist(rect2(2):rect2(2)+rect2(4) , rect2(1):rect2(1)+rect2(3) , :);
        % Average:
        x_twistw=mean(x_twistw,1); x_twistw=mean(x_twistw,2);
        x_twistw = x_twistw(:);
        % avoid singularity
        x_twistw(end) = x_twistw(end-1);
%         if x_twistw(end)<0.001 
%             x_twistw(end) = 0.001;
%         end
        
        % show plot
        disp(['------------------------------------------------------------']);
        disp(['(' num2str(ind) ') Reference White Measurement (wavelength[nm]; signals): ']);
        disp(['rect: ' num2str(rect2)]);

        % for real measurements
        for k=1:size(wvls2b,2)
            if x_twistw(k) < 0
                x_twistw(k) = 0;
            end
            disp([num2str(floor(wvls2b(k))) '; ' num2str(x_twistw(k))]);
        end    
        disp(['Aver. sum of energy: ' num2str(sum(x_twistw))]);
        %-----------------------------------------------------

        fh3 = figure(3);
        set(fh3,'Name','Radiance');
        clf(fh3); 

        set(gcf,'color','white');   
        hold on;
        plot(wvls2b,x_twistw,'.','MarkerEdgeColor','r');    % show real measurements
        plot(wvls2b,x_twistw, '--b');    % raw measurement

        % sending the data to output
        refwhite = x_twistw;
        whitefn = regexprep(sourcepath, '.mat', '_refWhite.mat','ignorecase');
        save(whitefn,'refwhite');

        xlabel('Wavelength[nm]');
        ylabel('Relative radiance');
        title('Spectral Power Distribution');

        % determine the maximum in the axes
        if (max(x_twistw)>sclr)
            sclr = max(x_twistw); % the most
        end
        ylim([0,sclr]);

        xlim([300,1010]);

        disp('Calculating reflectance from radiance...');
        
        sRGB = R2sRGB(wvls2b,x_twist,refwhite, islinear, isstretch);        
        % calculating reflectance (white balancing)
        wx_twist = Rad2Ref(x_twist,refwhite);

        disp('Done.');
        %-------------------------------------------------------------------
    else
        disp 'this is the radiance mode.'
        wx_twist = x_twist;
        sRGB = R2sRGB(wvls2b,x_twist,0,islinear, isstretch);
    end
    
    %-------------------------------------------------------------------

    %================================================
    % Display Color Image
    %================================================
%     output(:,1) = wvls2b;
%     while 1
%         [sclr,x_twisti] = toshow(wx_twist,wvls2b,sclr,sRGB,ind,refwhite);
%         ind = ind + 1;        
%         output(:,ind+1) = x_twisti;
%         if (sclr == 0) 
%             output = output(:,1:(end-1));
%             save('measurement.mat', 'output');
%             if checkfigs(3)
%                close(3);
%             end
%             return;
%         end
%     end
end

function [sclr,x_twisti] = toshow(x_twist,wvls2b,sclr,sRGB,ind,refwhite)
    rect_post = 0;
    x_twist2=0;
    x_twisti=0;
    %================================================

    %================================================
    % Display one image data
    %================================================
    % figure
    % Multiple pts:
    fh1 = figure(1);
    set(fh1,'Name','Visualization');
    clf(fh1); 
    sumimg = sum(x_twist,3);
    
    [y2 rect2]=imcrop(sRGB,[]);
    if isempty(rect2)
        sclr = 0;
        return;
    end
        
    rect2=round(rect2);
    x_twist2=x_twist(rect2(2):rect2(2)+rect2(4) , rect2(1):rect2(1)+rect2(3) , :);
    
    % Average:
    x_twist2=mean(x_twist2,1); x_twist2=mean(x_twist2,2);
    rect_post=rect2;

    x_twist2=squeeze(x_twist2);
    % eliminate the last noise
    x_twist2(end) = x_twist2(end-1);
    
    %-----------------------------------------------------
    % regularizating signal
%    cf = smoothsplinefitting(wvls2b,x_twist2);
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
%         x_twist2r(k) = cf(wvls2b(k));
%         temp = cf(wvls2i(k));
%         if temp<0
%             temp = 0;
%         end
%         x_twist2r(k) = temp;
% %         disp([num2str(floor(wvls2i(k))) '; ' num2str(x_twist2r(k))]);
%     end   
    
    disp(['------------------------------------------------------------']);
    disp(['(' num2str(ind) ') Measurement (wavelength[nm]; signals): ']);
    disp(['rect: ' num2str(rect2)]);
    
    % for real measurements
    for k=1:size(wvls2b,2)
        if x_twist2(k) < 0
            x_twist2(k) = 0;
        end
        disp([num2str(floor(wvls2b(k))) '; ' num2str(x_twist2(k))]);
    end    
    disp(['Aver. sum of energy: ' num2str(sum(x_twist2))]);
    %-----------------------------------------------------
    fh3 = figure(3);
    set(fh3,'Name','Radiometric Measurement');
    clf(fh3); 
    
    set(gcf,'color','white');   
    hold on;
    plot(wvls2b,x_twist2,'.','MarkerEdgeColor','r');    % show real measurements
    %plot(wvls2i,x_twist2r, '--b');    % interpolation
    plot(wvls2b,x_twist2, '--b');    % raw measurement

    % sending the data to output
    x_twisti = x_twist2;
    
    xlabel('Wavelength[nm]');
    title('Spectral Power Distribution');
    
    % determine the maximum in the axes
    if (max(x_twist2)>sclr)
%        sclr = 0.1;
        sclr = max(x_twist2); % the most
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
