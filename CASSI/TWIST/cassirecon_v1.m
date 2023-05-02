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
% Acknowlegments
% Portions of this file were based on the original code of David Kittle 
% (Duke University).
%============================================================================
% function cassirecon(type, rect, file_name_calib, file_name_main, starti)
function cassirecon_v1( type, rect, file_name_calib, file_name_main, starti )
% chopping=800 --> create the mini color chart crop at the center
% We have to crop the edge trim of the coded aperture.
    nopad = 1; % zero-padding off
    %nopad = 0; % zero-padding on

    close all;
    
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    IS_VCCASSI = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Twist Paramter sets
    %
    if nargin==0
%         disp 'enter the type: [uv] or [vis] or [ir]'
%         return;
        type = 'vccassi';
    end
    if nargin < 3
        file_name_calib = '';
        file_name_main = '';
    end
    %============================================%
    % Automatic parallelization for optimization %
    % note matrix volume and total memory is     %
    % the main criteria                          %
    %============================================%
    %totalmemory = 16; % birdcruncher.eeb.yale.edu
    %totalmemory = 12; % spectra.cs.yale.edu
    %totalmemory = 24; % Rick's machine 8-core
    totalmemory = 32;  % VCLAB's PCs
    
    %============================================%
    %     Parameters for wavelength              %
    %============================================%
    disp 'Main data reading done!';  
    % enter the wavelength
    laser_wl = 550;  %DON'T CHANGE THIS -> for KAIST
    % fixed by our filter set (given by Duke)
    
%     if strcmp(type,'all')==1 
%         %=====================================================================
%         % KG-3 filter 300-800nm -> needs to apply a transmittance correct
%         % function
%         % deltab=3; % 32 channels
%         % deltab=4; % 24 channels
%         % deltab=5; % 26 channels
%         % deltab=6; % 8 channels
%         deltab=7; % 6 channels
%         
%         start_wvl=350;
%         end_wvl=700;
%         %=====================================================================
%     elseif strcmp(type,'vis')==1 
%         deltab=3; % for 11 channels; [BEST-LDR] 3 is the best even with bugfixed
% %        deltab=2; % for  channels; [HDR] -> better than LDR, but more spectral noise was observed
% %        deltab=2; % for 21 channels; [CASSI calibration alignment failed]
% %        deltab=1; % for 31 channels; It failed with low image quality even with the bug-fixed version
%         %=====================================================================
%         % The CASSI double Amici Prism configuration limits this range to
%         % 450 and 650nm although the Baader filter range is wider as 430--680nm.
%         start_wvl=450; %DON'T CHANGE THIS (Baader)
%         end_wvl=650; %DON'T CHANGE THIS (Baader)
%         %=====================================================================
%     elseif strcmp(type,'uv')==1 
%         deltab=5; % [BEST] 5 makes 10bins(10nm) w/o TWIST crashing.
%         start_wvl=330; %DON'T CHANGE THIS (Baader)
%         end_wvl=380; %DON'T CHANGE THIS  (Baader)(no enery through the filter)
%         % end_wvl=450; % for the KG-3 filter not working well
    if strcmp(type,'uv')==1 
        % Filter 1 (blue):
        % Wavelength: 360-516nm
        % Bandwidth: 156nm
        % Parameters: 
        % Deltab: 3 -> 16 channels (359  364  369  375  382  389  397  405  415  425  437  450  464  480  497  516) <- about 10nm
        % actual:  359-516nm (157nm)
        % deltab=1, uv -> 359  361  362  364  366  367  369  371  373  375  377  379  382  384  386  389  391  394  397  399  402  405  408  412  415  418  422  425  429  433  437  441  445  450  454  459  464  469  474  480  485  491  497  503  509  516
        deltab=3;
        start_wvl=345; % 345: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER) (345 makes it start 359)
        end_wvl=516; % 516: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER)
    elseif strcmp(type,'vis')==1 
        % Filter 2 (yellow):
        % Wavelength: 514-700nm
        % Bandwidth: 186nm
        % Deltab: 1 -> 20 channels (516  523  530  537  544  552  560  568  577  586  595  604  614  624  635  645  657  668  680  692) <- about 10nm (threw away 516 here???too noisy)
%        deltab=2; % when deltab=1, reconstruction fails (shifting to the left)
        deltab=1; % THIS NEEDS SPECIAL OFFSET CORRECTION
        start_wvl=514; % 514: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER)
        end_wvl=710; % 710: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER) (710 makes it start 705)
    elseif strcmp(type,'ir')==1 
        % Filter 3 (ir):
        % Wavelength: 680-1000nm
        % Bandwidth: 380nm
        % Deltab: 1 -> 20 channels (516  523  530  537  544  552  560  568  577  586  595  604  614  624  635  645  657  668  680  692) <- about 10nm (threw away 516 here???too noisy)
%        deltab=2; % when deltab=1, reconstruction fails (shifting to the left)
        deltab=1; % THIS NEEDS SPECIAL OFFSET CORRECTION
        start_wvl=680; % 680: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER)
        end_wvl=1010; % 1010: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER) (1010 makes it start 1002)
    elseif strcmp(type,'nofilter')==1 
        % Cu (# of Channels): 84
        % Start wv: 359.48
        % End wv: 1025.8948
        % Spectrum bins: 359   361   362   364   366   367   369   371   373   375   377   379   382   384   386   389   391   394   397   399   402   405   408   412   415   418   422   425   429   433   437   441   445   450   454   459   464   469   474   480   485   491   497   503   509   516   523   530   537   544   552   560   568   577   586   595   604   614   624   635   645   657   668   680   692   705   718   732   746   761   776   791   807   824   841   859   878   897   916   937   958   980  1002  1025
        deltab=1; 
        start_wvl=345;  % 345: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER) (345 makes it start 359)
        end_wvl=1010; % 1010: DON'T CHANGE (PHYSICAL PROPERTY OF FILTER) (1010 makes it start 1002)
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif strcmp( type, 'vccassi' ) == 1
%         deltab = 3;
        deltab = 6;
%         start_wvl = 400; % for Hoya IR filter
%         end_wvl = 700; % for Hoya IR filter
%         laser_wl = 550;
        start_wvl = 450;
        end_wvl = 720;
        laser_wl = 550;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        disp 'enter the type: [uv] or [vis] or [ir]'
        return;
    end        


    %=====================================================================
    % starting iteration
    if nargin < 5 % for visible spectrum
        starti = 1;
    end
    
    %=====================================================================
    % ask filename for the calibration data
    if (isempty(file_name_calib))
        % file read
        %[file_name,file_path] = uigetfile({'*.bmp';'*.hdr';'*.*'},'Choose a calibration file');
        [file_name_calib,file_path_calib] = uigetfile({'*.png';'*.hdr';'*.*'},'Choose a calibration file');
        % leave function when no file is selected.
        if size(file_path_calib,2) == 1
            return;
        end
    else
        file_path_calib = [pwd '/'];             
    end
    sourcecalib = char(strcat(file_path_calib,file_name_calib));       
    
    [pathstr, name, ext] = fileparts(sourcecalib);
    if strcmp(ext,'.hdr')
        cdata = readhdr(sourcecalib);
        cdata = cdata(:,:,2);
    else
        cdata = single(imread(sourcecalib));
    end
    
    cdata = stretch_hist(cdata);
    
    % read the size of cdata
    height = size(cdata,1);
    width = size(cdata,2);
        
    disp 'Calibration data reading done!';
    %=====================================================================
    % ask filename for the main listfile
    if (isempty(file_name_main))
        % file read
        [file_name_main,file_path_main] = uigetfile({'*.mat','mat';},'Choose a main capture mat file');
        % leave function when no file is selected.
        if size(file_path_main,2) == 1
            return;
        end
    else
        file_path_main = [pwd '/'];          
    end
    sourcepath = char(strcat(file_path_main,file_name_main));  
    
    % load the source MAT file
    load(sourcepath);

    %=====================================================================
    % total resolution of one side of the square sensor
    totalres = size(cdata,1);
    
    if strcmp( type, 'vccassi' ) == 0
        %=====================================================================
        % chopping edge pixels (removing the trim of coded aperture)
        % chopping 900 makes 249x249 resolution image. It takes only 4 minutes
        % default should be 50 in order to remove the black trim of the coded
        % aperature.
    %    chopping = 48; % DEFAULT should be 48
        chopping = 0; % DEFAULT should be 48
        cdata = cdata((1+chopping):size(cdata,1)-chopping,(1+chopping):size(cdata,1)-chopping,:);
        I_trans = I_trans((1+chopping):size(I_trans,1)-chopping,(1+chopping):size(I_trans,1)-chopping,:);

        % Chopping source edge data
        %img_c = uint8(cdata(:,:));
        img_i = uint8(I_trans(:,:,1));
        %figure; imshow(img_c); set(gcf,'name','Coded aperture');

        %=====================================================================
        % with default chopping
        % these rect values are locations in 2024 x 2024
        if nargin < 2;
            % launch the selection UI
            [X crect] = selectregion(img_i);

            rect(1) = crect(1) + chopping; % x
            rect(2) = crect(2) + chopping; % y
            rect(3) = crect(3); % width
            rect(4) = crect(4); % height

            % segment the data
            cdata = cdata(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),:);
            I_trans = I_trans(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),:);
            clear('img_i', 'X');
        elseif size(rect,2) == 4 % use given data (after chopping)
            crect(1) = rect(1) - chopping;
            crect(2) = rect(2) - chopping;
            crect(3) = rect(3);
            crect(4) = rect(4);

            cdata = cdata(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),:);
            I_trans = I_trans(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),:);
        else
            return;
        end
        %=====================================================================
        % After cropping the cdata and I_trans
        % we need to account for the chopping offsets in rect(1) and rect(2)
        disp(['========================================']);
        disp([sprintf( '[Selected region in %dx%d]', height, width )]);
        disp(['Rect (x,y,w-1,h-1):' num2str(uint16(rect))]);
        %=====================================================================
    else
        % this is for KAIST
            img_i = uint8(I_trans(:,:,1));
            
            % launch the selection UI
            [X crect] = selectregion(img_i);

            rect(1) = crect(1); % x
            rect(2) = crect(2); % y
            rect(3) = crect(3); % width
            rect(4) = crect(4); % height

            % segment the data
            cdata = cdata(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),:);
            I_trans = I_trans(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),:);
            clear('img_i', 'X');
    end
    
    
    % show cropped results
    figure; imshow(uint8(cdata(:,:)));set(gcf,'name','Coded aperture');
    figure; imshow(uint8(I_trans(:,:,1)));set(gcf,'name','Dispersion capture');
    %------------------------------------------------------------------

    %============================================%
    %     Parameters for reconstruction          %
    %============================================%
    %------------------------------------------------------------------
    % denoising calculatin (smoothing factor parameters) in a single
    % iteration.
    %------------------------------------------------------------------
    %taub=12; % too soft
    %taub=0.1; % DUKE settings: for higher resolution image --> it shoul be smaller.
    %taub=0.05; % for HDR still too high
    %taub=0.02; % still too sharp
    %taub=0.1; % BEST CHOICE (with Yale EEB Deploy)
    %taub=0.001; % Yale EEB: too sharp (2011-09-29)
    %taub=0.01; % Yale EEB: still rough, similar to 0.01
    %taub=0.05; % Yale EEB: Looks good but little bit risky for dark noise
%    taub=0.1; % BEST CHOICE (with Yale EEB Deploy) (INCHANG: ORIGINAL VAL)
     taub = 0.001;
    
    %------------------------------------------------------------------
    % piterb is a number of iteration in a single denoise operation.
    % for tvdenoise.m fuction
    %------------------------------------------------------------------
    % piterb=2; % 
%     piterb=8; % took double time than piterb=4, but no improvement in quality
%     piterb=40; % too soft
    piterb=4; % Default: Yale EEB Deploy Final % this was the best with cassidenoise.m
    %------------------------------------------------------------------

    %------------------------------------------------------------------
    % Maximum TWIST iteration (from Ver. 5)
    %------------------------------------------------------------------
    % for original CASSI
    %iterationsb=40; % after 40 almost same... => Currently, the best (1hr40min)
%    iterationsb=100; % [Good with Baader] 40 produce serious artifact with characterization => It makes no difference but time (3hours)
    iterationsb=200; % reasonable maximum
    %------------------------------------------------------------------

    %------------------------------------------------------------------
    % Iteration stop criteria in TWIST: tolerance to stop iterations. 
    %------------------------------------------------------------------
    % (the smaller, the more accurate)
    %tolA = 1e-4; % it makes it stop around 40
    %tolA = 1e-6; % it makes it stop around 50--100 (FINAL-HDR)
    tolA = 1e-8; % this is important to keep for iterating higher number (unlimited iteration) -> make no sense
    %------------------------------------------------------------------

    %============================================%
    % The end of parameters for reconstruction   %
    %============================================%
    
    % selected region volume
    regionheight = rect(4)+1;
    regionwidth = rect(3)+1;
    regiondepth = size(I_trans,3);
    regionvolume = regionheight * regionwidth * regiondepth;
    
    % a linear scaler

    if strcmp(type,'nofilter')==1     
        piecevolume_thres = 780 * 2024 * 6; % only for paper writing.
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif IS_VCCASSI
        piecevolume_thres = 780 * 2024 * 27; 
%         piecevolume_thres = 780 * width * 38; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else % ordinary cases
    %    piecevolume_thres = 780 * 2024 * 55; % measured by experiments (can be different)
        piecevolume_thres = 780 * 2024 * 31;%35; % 11/18/2011 (for Semrock three bands)
    end
    
    % a parallel piece
    % test for pieceheight % (a)dynamic parameter depends on the volume size of the region
    pratio = (regionvolume * 16) / (piecevolume_thres * totalmemory); % for Semrock UV 16
    
    if regionvolume <= piecevolume_thres
        pieceheight = regionheight;
    else
        pieceheight = floor(regionheight / pratio); 
    end
    piecewidth = regionwidth;
    piecedepth = regiondepth;
    piecevolume = pieceheight * piecewidth * piecedepth;

    % vertical overlap in segmentation
    pieceoverlap = 25;% (b)25px vertical overlap - fixed
    
    if pratio > 1.0
        inum = ceil((regionheight-2*pieceoverlap)/(pieceheight-2*pieceoverlap));
    else
        inum = 1;
        %a = 1;
        pieceheight = regionheight;
        pieceoverlap = 0;
    end
    disp(['========================================']);
    disp(['[Parallelization]']);
    disp(['Total number of parallel pieces: ' num2str(inum)]);
    disp(['Piece volume (h,w,d):' num2str(pieceheight) 'x' num2str(piecewidth) 'x' num2str(piecedepth)]);
    
%     %------------------------------------------------------------------
%     % segmentation parameters
%     %------------------------------------------------------------------
%     % one piece height
%     
%     % this is a reasonable level for 16GB system 
%     %---------------------------------
%     % a and b should be optimized w.r.t. the size of memory of the system
%     % merge_segments should be updated as well.
%     %---------------------------------
%     %a = 256; % this is good for deltab = 3 (for VIS)
%     % a = 220;% for deltab = 1 (for VIS)But image quality was bad
%     %a = 420; % this is good for deltab = 3 (for VIS)
% %    a = 600; % this is good for deltab = 3 (for VIS) - total 4 pieces (max. of memory usage is 15.3GB)
% %    a = 800; % for HDR 30 shifts (deltab=2)
%     if strcmp(type,'all')==1 
%         a = 300; % for HDR (KG-3) filter with deltab = 3 (32 channels);
%     else % for 'vis' and 'uv'
%         if (size(I_trans,3)<=30)
%             a = 1100; % for HDR 30 shifts (deltab=3) -> two pieces
%         else
%             a = 780; % for HDR 60 shifts (deltab=2) -> three pieces
%         end
%     end
% %    a = 2048+25; % for HDR 30 shifts (deltab=3) -> FOR ONE SHOT
%     %b = 32;
%     b = 25; % slightly dangerous...
%     %------------------------------------------------------------------    
    %=====================================================================
    % Displaying parameters:
    disp(['========================================']);
    disp(['[IMAGE_SIZE]']);
    disp(['Height: ' num2str(size(I_trans,1)) '/' num2str(totalres)]);
    disp(['Width: ' num2str(size(I_trans,2)) '/' num2str(totalres)]);
    disp(['']);
    disp(['[VARIABLES]']);
    disp(['Spectral Type: ' type]);
%    disp(['Piece Depth: ' num2str(piecedepth) 'px']);
    disp(['taub: ' num2str(taub)]);
    disp(['piterb: ' num2str(piterb)]);
    disp(['iterationsb: ' num2str(iterationsb)]);
    disp(['tolA: ' num2str(tolA)]);
    disp(['deltab: ' num2str(deltab)]);
    disp(['start wvl: ' num2str(start_wvl)]);
    disp(['end wvl: ' num2str(end_wvl)]);
    disp(['starting index: ' num2str(starti)]);
    disp(['========================================']);
    disp(['[START]']);
    %=====================================================================
    %=====================================================================
    % first derive source matrices (Cu and I_trans).

    % loading calibration data (fixed by hardware configuration)
    %[mono_input calcube]=laser_calcube2(laser_wl, cdata, handles);
    %laser_wl=534.3;
    
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp( type, 'vccassi' ) == 1
%     load vccassi_calib_data.mat
    load vccassi_calib_data_2.mat
else
    load wave_disp_cf3.mat % hardware calibration data from Duke   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Find blue (right)
    blue = abs( round( disp_fit( start_wvl ) - disp_fit( laser_wl ) ) );
%     Red (left, subtract)
   red = -abs( round( disp_fit( end_wvl ) - disp_fit( laser_wl ) ) );
  %%%%%%%%%%%%%%%%%%%%%%%
  % Reverse
  %%%%%%%%%%%%%%%%%%%%%%%
%     blue = -abs( round( disp_fit( start_wvl ) - disp_fit( laser_wl ) ) );
%     % Red (left, subtract)
%     red = abs( round( disp_fit( end_wvl ) - disp_fit( laser_wl ) ) );
    
    % for color cube
    calcube=zeros(size(cdata,1),size(cdata,2),length(red:blue),'single');
%     calcube=zeros(size(cdata,1),size(cdata,2),length(blue:red),'single');
    ind2=1;
%    for ind=blue:-2:red % this cannot make image for VIS
    for ind=blue:-1:red % why is it -2?
%     for ind=red:-1:blue % why is it -2?
        %calcube(:,:,ind2)=circshift(cdata,[0,ind]);% DUKE --> THIS IS WRONG DIRECTION as we change the capture code!!!
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(WRONG)        calcube(:,:,ind2)=circshift(cdata,[0,-ind]);% Min's try to change the dispersion direction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
%(FIXED) Don't disperse the coded pattern here. Instead, disperse the
%maksed image later in the function named 'R2()'!
        calcube( :, :, ind2 ) = cdata;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        ind2=ind2+1;
    end
    
    % wvls=linspace(425,750,size(calcube,3));
%    wvls=wvls_fit(disp_fit(550)-blue:disp_fit(550)-red); % by Duke
    wvls=wvls_fit(disp_fit(laser_wl)-blue:disp_fit(laser_wl)-red);% by Min
%     wvls=wvls_fit(disp_fit(laser_wl)-red:disp_fit(laser_wl)-blue);% by Min
    %wvls=disp_fit(disp_fit(425)-(0:length(blue:red)-1) )+7;

    disp('Calcube Loaded...');
    disp(['Calcube: ' num2str(size(calcube,1)) 'x' num2str(size(calcube,2))...
        'x' num2str(size(calcube,3))]);
    
%{
    % this is from old CASSI
    shiftb=[-shift(1,:);shift(2,:)];
    % Since +movement is to the left, invert horizontal
    shiftb(1,:)=shiftb(1,:); % - with/after 9/10
    % % no need to invert vertical since lens does this:
    shiftb(2,:)=-shiftb(2,:);  % + prior to 9/8
    % Just in case round-off-errors:
    shiftb=round(shiftb);
%}    
    %=====================================================================
    % Min's CASSI
    % shift(1) is x and shift(2) is y
    % since I changed gig_simple.m as below:
    % (duke) rot90(handles.activex1.GetRawData,1);
    % (Min) fliplr(rot90(handles.activex1.GetRawData,1));
    % I have to change shift direction from left to right!
    % (1) spectral dispersion shift direction -> (Min's) calcube(:,:,ind2)=circshift(cdata,[0,-ind]);
    % (2) shift(1) direction: (duke)-shift(1,:) -> (Min's) shift(1,:);
    shiftb=[ shift(1,:); -shift(2,:) ]; % case1: current version (12/28/2011)
    %shiftb=[shift(1,:);shift(2,:)]; % case2: worse
    %shiftb=[-shift(1,:);shift(2,:)]; % case3: much worse
    %shiftb=[-shift(1,:);-shift(2,:)]; % case4: worse
    %-------------------------------------
    shiftb=round(shiftb); % case1: current version (12/28/2011)
% ||A x - y ||_2 = 2.525e+003
% ||x||_1 = 5.660e+003
% Objective function = 1.440e+003
% Number of non-zero components = 216783
% CPU time so far = 1.137e+002    
    %-------------------------------------
    % scalar test (12/28/2011)
    %scalar = 1.1;% case5: worse than case1
    %scalar = 0.9;% case6: slightly worse than case1
    %scalar = 0.95;% case7: slightly better than case1
    %scalar = 0.98;% case8: slightly worse than case1
    %scalar = 0.91;% case10: slightly worse than case1
    %scalar = 0.94;% case11: slightly more better than case1
    %scalar = 0.92;% case11: slightly more better than case1
%-------------------------------------------------------
%     scalar = 0.93;% case9: slightly more better than case1 ==> This doens't work!!!!! ==> Go back to the original!!!!
%     shiftb=round(double(shiftb)*scalar); % New version from (12/28/2011)
    % [MIN] this means Piezo movement is not perfectly calibrated!
% Results:
% ||A x - y ||_2 = 1.827e+003
% ||x||_1 = 5.686e+003
% Objective function = 1.063e+003
% Number of non-zero components = 216783
% CPU time so far = 1.139e+002    
%-------------------------------------------------------
    %=====================================================================
    % Spectral Subsampling for efficiency 
    % I think VIS and UV should be done differently.
    % we need to increase the number for VIS and decrease for UV.
    %
    % below part reduces spectral resolution
    ind2=1;
    wvls2=zeros(1,round(size(calcube,3)/deltab),'single'); 
    % [Min] it should be single for memory efficiency (no benefit for
    % double), this is simple BW mask
    %Cu=zeros(size(calcube,1),size(calcube,2),round(size(calcube,3)/deltab)); % original
    Cu = zeros(size(calcube,1),size(calcube,2),round(size(calcube,3)/deltab), 'single'); % by Min
    
    %-----------------------------------------------------------
    % This is the old Non-linear sampling of spectra (from Duke)
    for ind=1:deltab:size(calcube,3)
        wvls2(ind2)=wvls(ind);
        Cu(:,:,ind2)=calcube(:,:,ind);
        ind2=ind2+1;
    end
    %-----------------------------------------------------------
    % This is the new uniform spectral sampling code (by Min)
    % 
    % TODO
    %-----------------------------------------------------------

    % Reduce to specified wavelength:
    start_wvl=find(wvls2<=start_wvl); if ~isempty(start_wvl), start_wvl=start_wvl(end); else start_wvl=1; end
    end_wvl=find(wvls2>=end_wvl);  if ~isempty(end_wvl), end_wvl=end_wvl(1); else end_wvl=length(wvls2); end
    wvls2b=wvls2(start_wvl:end_wvl);
    Cu=Cu(:,:,start_wvl:end_wvl);
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if IS_VCCASSI
       % (Min) blue is right, red is left
         disperse = int32(round( disp_fit( laser_wl ) -  disp_fit( wvls2b ) ));
   end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp('Done Building data set!');
    disp(['Cu (# of Channels): ' num2str(size(Cu,3))]);
    disp(['Start wv: ' num2str(wvls2(start_wvl))]);
    disp(['End wv: ' num2str(wvls2(end_wvl))]);
    disp(['Spectrum bins: ' num2str(floor(wvls2b))]);
    
    % After subsampling, delete the original 'calcube' and 'cdata'
    % for acquiring more memory space.
    clear ('calcube');
    clear ('cdata');

    % normalizing the inputs, Cu and I_trans
    % [Min] (2011-09-28) This scaling should be done for LDR and HDR! -->
    % --> this is very much affective on the reconstruction quality
    Cu = single(Cu)./255.; % Cu -> Coded Aperture
    I_trans = single(I_trans)./255.; % I_trans -> Shifted Images
%    Cu = single(Cu)./max(max(max(Cu))); % Cu -> Coded Aperture
%    I_trans = single(I_trans)./max(max(max(I_trans))); % I_trans -> Shifted Images

    idtttime = tic;

    %=====================================================================
    % each piece calculation
    % file list
    filenamelist = ['filelist_rec.txt'];
%        filenamelist = regexprep(filenamelist, '_rec.txt', ['_' num2str(floor(wvls2b(j))) 'nm_rec.txt']);    

    if nargin<5
        delete filenamelist;
        filelist = fopen(filenamelist,'w');
    else
        filelist = fopen(filenamelist,'a');
    end

    %=====================================================================
    % actual height for each segmentation
%     if size(I_trans,1)*size(I_trans,2)<=2048*8
%         inum = 1;
%         a = 1;
%         b = 0;
%     else
%         inum = ceil((regionheight-2*b)/(a-2*b));
%     end
%     disp(['total number of segments: ' num2str(inum)]);
    % total height = 2048 x 8;
    % a = 256
    % b = 32 %(overlap)
    % index       compute range                                    sample range
    % n=1               1 ~ a                                       1 ~ a-b
    % n=2       (n-1).a - (n-1).2.b ~ n.a - (n-1).2.b   (n-1).a - ((n-1).2-1).b ~ n.a - ((n-1).2+1).b
    % until n=11   n.a - (n-1).2.b <= 2048
    % n=11                              ~ 2048                          ~ 2048
    %=====================================================================
    % spatial parallelization loop
    if inum==1
        disp(['========================================']);
        disp(['Spatial Patch Index: 1/1']); 
        
        Cup = Cu(:, :, :); % multi-spectral
        I_transp = I_trans(:,:, :);

        % run iteration
        piececalculation(1, sourcepath, filelist, Cup, I_transp,...
        disperse, iterationsb, shiftb, taub, piterb, deltab, wvls2b, tolA);        
    else
        for i=starti:inum
            disp(['========================================']);
            disp(['Spatial Patch Index: ' num2str(i)]); 

            if i==1
                stcom = 1;
                edcom = pieceheight;
                stsam = 1;
                edsam = pieceheight - pieceoverlap;
            elseif i==inum
                stcom = (i-1)*pieceheight - (i-1)*2*pieceoverlap+1;
                edcom = regionheight;
                stsam = (i-1)*pieceheight - ((i-1)*2-1)*pieceoverlap+1;
                edsam = regionheight;
            else            
                stcom = (i-1)*pieceheight - (i-1)*2*pieceoverlap+1;
                edcom = i*pieceheight - (i-1)*2*pieceoverlap;
                stsam = (i-1)*pieceheight - ((i-1)*2-1)*pieceoverlap+1;
                edsam = i*pieceheight - ((i-1)*2+1)*pieceoverlap;
            end            

            disp(['stcom: ' num2str(stcom)]);
            disp(['edcom: ' num2str(edcom)]);
            disp(['stsam: ' num2str(stsam)]);
            disp(['edsam: ' num2str(edsam)]);
            % separating an image
            Cup = Cu(stcom:edcom, :, :); % multi-spectral
            I_transp = I_trans(stcom:edcom,:, :);

            % run iteration
            piececalculation(i, sourcepath, filelist, Cup, I_transp,...
            iterationsb, shiftb, taub, piterb, deltab, wvls2b, tolA);
        end % spetial
    end
    % close list file
    fclose(filelist); % it doesn't work (maybe matlab bug)

    tttime = toc(idtttime);
    disp('Total reconstruction finished');
    disp(['Time: ' formattime(tttime)]);  
%    disp(['Time: ' num2str(tttime)]); 

    fclose('all');        

    % merge individual calculation into one
    %merge_segments(type,filenamelist);
    merge_segments(type, filenamelist, pieceheight, pieceoverlap, rect, totalres, tttime, nopad);
    %merge_channels(type,fnreconfilenamelist); % this is totally wrong

%{
    for j=1:size(Cu,3)
        disp(['Spectral Band Index: ' num2str(j)]); 
        disp(['Wavelength: ' num2str(floor(wvls2b(j))) 'nm']);   
        % file list
        %filenamelist = ['filelist_rec.txt'];
        %filenamelist = regexprep(filenamelist, '_rec.txt', ['_' num2str(floor(wvls2b)) 'nm_rec.txt']);    
        
        % merge calculated data
        %merge_segments(type,filenamelist);
        

%         %==========================================================
%         % merge calculated data
%         if strcmp(type,'vis')==1 
%             merge_segments; %'filelist_rec.txt'
%         elseif strcmp(type,'uv')==1 
%             merge_segments_uv; %'filelist_rec.txt'
%         end   
%         %==========================================================
    end
%}  
    msgbox('Done!','Message','warn');beep;
end


%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function piececalculation(index, sourcepath, filelist, Cup, I_transp,...
%     iterationsb, shiftb, taub, piterb, deltab, wvls2b, tolA)
function piececalculation(index, sourcepath, filelist, Cup, I_transp,...
    disperse, iterationsb, shiftb, taub, piterb, deltab, wvls2b, tolA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    iterationsb, shiftb, taub, piterb, deltab, wvls2b, reconfilelist)


    idttime = tic;
    %[x_twist_orig dummy]=RUNME_calcuberecon2(Cu2, y2, iterationsb, shiftb, taub, piterb,rect2crop);
    
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [x_twist] = RUNME_calcuberecon_min(Cup, I_transp, disperse, iterationsb, shiftb, taub, piterb, tolA);
%     [x_twist] = RUNME_calcuberecon_min(Cup, I_transp, iterationsb, shiftb, taub, piterb, tolA); % by Min
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %[x_twist dummy]= MKL_calcuberecon(Cup, I_transp, iterationsb, shiftb, taub, piterb, pathname); % by Hongzhi
    ttime=toc(idttime); 
      
    disp('Reconstruction of a piece was finished');
    disp(['Time: ' formattime(ttime)]);  
    
    %=====================================================================
    % displaying final => this could be wrong!
    % deltab decides the depth of spectral intervals.
    % This part can be used to solve the misalignment problem of spectral
    % wavelength
%     cal2 = zeros(size(x_twist),'single'); 
%     xs = size(x_twist,2); ys = size(x_twist,1); mid = round( size( x_twist,3 )/2 );
%     zs = size(x_twist,3); % number of spectrum
%     for ind=1:zs
% %[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         if ~isempty( disperse )
%             cal2(:,:,ind)...
%                  =circshift( x_twist(:,:,ind), [ 0 -disperse( ind )] ); % (Min) this sould be inverse of disperse(ind)!
% %                = circshift( x_twist(:,:,ind), [ 0 disperse( ind )] );
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         elseif deltab==1 % THIS IS FOR LONGER WAVELENGTH (YELLOW CHANNEL)
%             sshift = -((zs-1)*2-(ind-1)*2)+zs; % 
%             %sshift = 2*((ind-1)-mid); % [Min] This is working for YELLOW - 2011/11/11 % FINAL
%             cal2(:,:,ind)=circshift(x_twist(:,:,ind),[0 sshift]);
%         else % when deltab is higher than 2, circshift is not necessary [Min] 2011/11/11 % Final
%             cal2=x_twist;
%         end
%     end
% %[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     if ~isempty( disperse ) && deltab == 1 % THIS IS FOR LONGER WAVELENGTH (YELLOW CHANNEL)
% %     if deltab==1 % THIS IS FOR LONGER WAVELENGTH (YELLOW CHANNEL)    
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         cal2=circshift(cal2,[0 zs]); % put it back
%     end
%     x_twist = cal2;
    %=====================================================================
    % Save data as a file
    %savefilename = regexprep(sourcepath, '.mat', ['_' num2str(floor(wvls2b)) 'nm_' num2str(index) '_rec.mat']);    
    %savefilename = regexprep(sourcepath, '.mat', ['_' num2str(index) '.mat']);    
    [pathstr, name, ext] = fileparts(sourcepath);
    name = regexprep(name, 'merged_', ['recon_' num2str(index) '_']);
    savefilename = [pathstr, '/', name, ext];

   
    save(savefilename,'shiftb','taub','piterb','Cup', 'iterationsb',...
        'wvls2b','deltab','x_twist','ttime')

    % write the filename into main list file
%    fprintf(reconfilelist, '%s\n', savefilename); % write filename   
    
    
    disp('Done saving data');      

    % save filename
    fprintf(filelist, '%s\n', savefilename); % write filename    
end

function [x_twist_orig]= MKL_calcuberecon(Cu, cdata, its, shift,tau,piter, pathname)% edited by Min
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
write_matrix_bin([pathname 'Cu.bin'], Cu);
write_matrix_bin([pathname 'cdata.bin'], cdata);
write_matrix_bin([pathname 'shift.bin'], shift);
write_param([pathname 'param.txt'], its, tau, piter);

%!C:\Scanning_System\ImagingSystem\CPUAccelerated on Code_Feb_2010\Simulation\twist
%!C:\Scanning_System\Cassi_Imaging_Pipeline\Cassi_Program\rec_gui\c_code\twist\Debug\twist
%!C:\Scanning_System\Cassi_Imaging_Pipeline\Cassi_Program\rec_gui\c_code\twist\Release\twist
!twist.exe

x_twist_orig = read_matrix_bin([pathname 'x_twist.bin']);

end

% close all;clear all;clc;

%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x_twist_orig] = RUNME_calcuberecon_min(Cu, cdata, its, shift,tau,piter,tolA)
function [x_twist_orig] = RUNME_calcuberecon_min(Cu, cdata, disperse, its, shift,tau,piter,tolA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cu: Transformation tensor
% cdata: Sensored Images (W x H x n)
% its: Iteration number

% rot=zeros(size(shift));

% Image to process:
%y = double(cdata); % cdata is just BW floating data
y = cdata; % single precision should be enough
clear('cdata');

y = y.*(y>=0);
%y=reshape(y,[size(y,1),size(y,2),size(shift,2)]); % (Min) we don't need
%this line (technically no change on the matrix)

%y = flipdim(y,1);

n1 = size(y,1);
n2 = size(y,2);
nt=size(shift,2);
m = size(Cu,3);
% y = y/max(y(:)); 
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IGNORE THIS
% y=y(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Single precision:
%y=single(y);
%Cu=single(Cu);

% (Min) normalization? why? necessary? -> by test, without normalization
% this TWIST algorithm failed. So we have to fix this MAXIMUM y and Cu
% consistently.
%y=y/(max(y(:))*1);
%Cu=Cu/(max(Cu(:))*1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shifting measurement cube and building up 4D measurement cube data.
%tic
Cus=zeros(size(Cu,1),size(Cu,2),m,nt,'single');
for ind=1:nt
    Cint=circshift(Cu,[shift(2,ind),shift(1,ind),0]);
    Cus(:,:,:,ind)=Cint(1:size(Cu,1),1:size(Cu,2),:);
end
%toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% smoothing parameter (empirical)
% tau = .3;

% ------------  TV experiments ---------------------
% handle functions for TwIST
%  convolution operators

% A = @(x) Rfuntwistv2(x,n1,n2,m,Cu,shift,rot);
% AT = @(x) RTfuntwistv2(x,n1,n2,m,Cu,shift,rot);
%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = @(f) R2(f,n1,n2,m,Cus,nt, disperse);
AT = @(y) RT2(y,n1,n2,m,Cus,nt, disperse);
% A = @(f) R2(f,n1,n2,m,Cus,nt);
% AT = @(y) RT2(y,n1,n2,m,Cus,nt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% denoising function;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  New TV denoising code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% piter = 4;

% Psi = @(x,th) MyTVpsi(x,th,tau,piter,n1,n2,m,[1 1 0]);
Psi = @(x,th) cassidenoise( x, th, piter ); % from Duke
% Psi = @(x,th) tvdenoise(x,th,piter); % this function didn't work well (for reconstruction)

% Phi = @(x) MyTVphi(x,n1,n2,m,[1 1 0]);
% Phi = @(x) TVnormspectralimaging(x);
Phi = @(x) TVnorm3D(x);

%tolA = 1e-8; %DUKE

% -- TwIST ---------------------------
% stop criterium:  the relative change in the objective function
% falls below 'ToleranceA'

[x_twist_orig,dummy,obj_twist,times_twist,dummy,mse_twist]=TwIST( ...
    y,A,tau,...
    'AT', AT, ...
    'Psi', Psi, ...
    'Phi',Phi, ...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',its,...
    'ToleranceA',tolA,...
    'Debias',0,...
    'Verbose', 1 );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [INCHANG]
% x_twist_orig = reshape( x_twist_orig, n1, n2, m );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% temp = shiftCube(x_twist_orig);
% x_twist_orig = temp; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [sjjeon] NESTA solver test by Seokjun Jeon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mu = 0.2;
% opts = [];
% opts.TolVar = tolA;
% opts.stopTest = 1;
% opts.Verbose = 1;
% opts.TypeMin = 'tv';
% delta = Phi(y);
% 
% 
% [x_twist_orig]=NESTA( ...
%     A, AT, y, mu, delta, opts);
% 
% temp = shiftCube(x_twist_orig);
% x_twist_orig = temp; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% x_twist = flipdim(x_twist_orig,1);
% 
% x_twist = x_twist.*(x_twist>=0);
% x_twist = x_twist/max(x_twist(:));
% 
% dispCubeAshwin(x_twist,140,round(linspace(452,660,25)));


% x_twist = flipdim(x_twist_orig,1); %x_twist_orig; 
% x_twist = x_twist.*(x_twist>=0);
% x_twist = x_twist/max(x_twist(:));
% 
% dispCubeAshwin(x_twist,200,round(linspace(453,653,z_num)));

% figure(1000);
% plot(log(dummy),'r','LineWidth',2,'LineWidth',2)
% legend('TwIST')
% st=sprintf('tau = %2.2e',tau);  title(st)
% ylabel('Obj. function')
% xlabel('CPU time (sec)')
% grid on

% toc
end


function y = TVnorm3D(x)
% this is phi function (this produces the summation of the magnitudes of gradients)
% TVnonmspectralimging --> one constant
nt=size(x,3);
y=zeros(1,nt,'single');
z1=zeros(size(x,1),1,'single');
z2=zeros(1,size(x,2),'single');

%  the summation of (the magnitudes of gradients).^2
for ind = 1:nt
% original cassi: the summation of the magnitudes of gradients
%     y(ind) = sum(sum(sqrt(diffh(x(:,:,ind),z1).^2+diffv(x(:,:,ind),z2).^2)));
% revised by Min (the summation of the magnitudes of gradients)^2 
% on 4/5/2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [INCHANG] Use L1 norm!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   y(ind) = sum(sum((abs(diffh(x(:,:,ind),z1))+abs(diffv(x(:,:,ind),z2)))));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    y(ind) = sum(sum((diffh(x(:,:,ind),z1).^2+diffv(x(:,:,ind),z2).^2)));
% for some reason, it didn't work properly with some images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

y = sum(y);

end

% h=[0 1 -1];
function y=diffh(x,z1)
% z1=zeros(size(x,1),1);
x = [x z1]-[z1 x];
x(:,1) = x(:,end)+x(:,1);
y = x(:,1:end-1);
end

% h=[0 1 -1]';
function y=diffv(x,z2)
% z2=zeros(1,size(x,2));
x = [x; z2]-[z2; x];
x(1,:) = x(end,:)+x(1,:);
y = x(1:end-1,:);
end

% For normal cpu:

%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = R2( f, n1, n2, m, Cs, nt, disperse) % y = Ax (h*w*snap)
% function y = R2( f, n1, n2, m, Cs, nt) % y = Ax (h*w*snap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLD R2() implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Make sure in 4D shape:
% f=reshape(f,[n1,n2,m]);
% % Punch:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % [Min] Here multidimensional dot product was impremented 
% %       as elementary product plus sum 
% %       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% % Elementary product
% gp=repmat(f,[1 1 1 nt]).*Cs; % 4D * 4D
% % Sum up the 3rd dimensions in each seperate image:
% y=sum(gp,3); % 4D -> 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% y=y(:); % vectorize

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [INCHANG] New R2 implementation by INCHANG
% (1) Disperse the hyperspectral channels after masking the coded aperture
% (2) Shift the hyperspectral image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vector to Image
f = reshape( f, [ n1, n2, m ] );

% Apply masking by the coded aperture
gp = repmat( f, [1 1 1 nt] ).*Cs;

% shift the masked channels
for i = 1:m
    gp_1channels = gp( :, :, i, : );
    if isempty( disperse )
        gp( :, :, i, : ) = circshift( gp_1channels, [ 0, -i, 0 ] );
    else
        gp( :, :, i, : ) = circshift( gp_1channels, [ 0, disperse( i ) , 0 ] ); % (Min) blue must be right
    end
end
% Projection
y = squeeze( sum( gp, 3 ) );
% Vectorize
% y = y(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = RT2( y, n1, n2, m, Cs, nt, disperse ) % f = ATy (h*w*spec)
% function f = RT2( y, n1, n2, m, Cs, nt ) % f = ATy (h*w*spec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLD RT2() implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % this is for captured image (y) and Cs is coded aperture (Cu)
% % nt=size(shift,2);
% % Make sure it is in 4D shape
% % n1=rows, ie 480
% % n2=cols, ie 640
% % length(shift) = number of individual images
% y=reshape(y,[n1,n2,1,nt]);
% % Replicate to number of channels in system:
% yp=repmat(y,[1,1,m,1]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % [Min] Here multidimensional dot product was impremented 
% %       as elementary product plus sum 
% %       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% % Elementary product
% yp=yp.*Cs; % 4D * 4D
% % Sum up on the 4th dimension:
% f=sum(yp,4); % DUKE: 4D -> 3D
% f = f(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [INCHANG] New RT2 implementation by INCHANG
% (1) Disperse the hyperspectral channels after masking the coded aperture
% (2) Shift the hyperspectral image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vector to Image
y = reshape( y, [ n1, n2, 1, nt ] );
% y = reshape( y, [ n2, n1, 1, nt ] );
% Replicate
yp = repmat ( y, [ 1, 1, m, 1 ]);
% disperse
for i = 1:int32( m )
%     gp(:,:,i) = circshift( gp, [ 0, -i ] );
    yp_1channels = yp( :, :, i, : );
    if isempty( disperse )
        yp( :, :, i, : ) = circshift( yp_1channels, [ 0, i, 0 ] );
    else
        yp( :, :, i, : ) = circshift( yp_1channels, [ 0, -disperse( i ), 0 ] ); % (Min) the inverse of blue must be the inverse of right
    end
end
% Mask
yp = yp.*Cs;   
% Proejction
f = sum( yp, 4 ); % DUKE: 4D -> 3D
% Vectorize
% f = f(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function y = disp_cal_fw(x)
%     General model Power2:
%      f(x) = a*x^b+c
% Coefficients (with 95% confidence bounds):
%        a = -1.535e+005  (-1.597e+005, -1.473e+005)
%        b =      -1.236  (-1.244, -1.228)
%        c =       85.98  (85.45, 86.51)
% 
% Goodness of fit:
%   SSE: 882.7
%   R-square: 0.9995
%   Adjusted R-square: 0.9995
%   RMSE: 1.052
    a = -1.535e+005;%  (-1.597e+005, -1.473e+005)
    b =      -1.236;%  (-1.244, -1.228)
    c =       85.98;%  (85.45, 86.51)

    y = a.*x.^b+c;
end

function x = disp_cal_bw(y)
%     General model Power2:
%      f(x) = a*x^b+c
% Coefficients (with 95% confidence bounds):
%        a = -1.535e+005  (-1.597e+005, -1.473e+005)
%        b =      -1.236  (-1.244, -1.228)
%        c =       85.98  (85.45, 86.51)
% 
% Goodness of fit:
%   SSE: 882.7
%   R-square: 0.9995
%   Adjusted R-square: 0.9995
%   RMSE: 1.052
    a = -1.535e+005;%  (-1.597e+005, -1.473e+005)
    b =      -1.236;%  (-1.244, -1.228)
    c =       85.98;%  (85.45, 86.51)
    x = exp(1/b * log((1/a)*(y-c)));
end