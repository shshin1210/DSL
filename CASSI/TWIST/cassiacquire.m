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
% function varargout = cassiacquire(varargin)
function varargout = cassiacquire(varargin)
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @cassiacquire_OpeningFcn, ...
                   'gui_OutputFcn',  @cassiacquire_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

%====================================================================
% Initialization Callback
% Start up functions
% --- Executes just before cassiacquire is made visible.
function cassiacquire_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<*INUSL>
% Choose default command line output for cassiacquire
handles.output = hObject;

%----------------------------------%
% Min's initial processes
%----------------------------------%
%clc;
handles.pathname = [pwd '\'];
set(handles.directory,'String', handles.pathname); 
% initializing camera setting
handles.activex1.Rotate=180;
handles.activex1.Flip = 0;
handles.activex1.Magnification = 0;

handles.activex1.GainAbs = 15;

updateHistogram(handles);


% start camera
%startcamera(handles); % not working

% set autoexposure
% exp_global_Callback(hObject, eventdata, handles);

% autoexposure(hObject,handles); %start camera need times % not working
%----------------------------------%

% Update handles structure
guidata(hObject, handles);

% piezo initialize
piezo_init_Callback(hObject, eventdata, handles);

% setPreview(handles);
% UIWAIT makes cassiacquire wait for user response (see UIRESUME)
% uiwait(handles.figure1);
%====================================================================

%====================================================================
% --- Outputs from this function are returned to the command line.
% Ending Callback
function varargout = cassiacquire_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;
endcamera(handles);
%====================================================================

% function setPreview(handles)
% plot(handles.axes_preview,0,0); H = gca;
% set(H, 'XTick',[0:1], 'XTickLabel',[])
% set(H, 'YTick',[0:1], 'YTickLabel',[])
% set(gca, 'xminortick','off') 
% set(gca, 'yminortick','off') 
% %imshow('cameraman.tif');
% %img=fliplr(rot90(handles.activex1.GetRawData,1));
% %img = (double(img)./max(max(double(img)))).^(1/2.2); % gamma correction
% %imshow(img);
% 
% function updatePreview(handles)
% %imshow('cameraman.tif');
% plot(handles.axes_preview,0,0); H = gca;
% img=fliplr(rot90(handles.activex1.GetRawData,1));
% img = (double(img)./255).^(1/2.2); % gamma correction
% imshow(img);

function num_frames_Callback(hObject, eventdata, handles)
% Hints: get(hObject,'String') returns contents of num_frames as text
%        str2double(get(hObject,'String')) returns contents of num_frames as a double
global num_translations
num_translations=str2double(get(hObject,'String'));


% --- Executes during object creation, after setting all properties.
function num_frames_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function setHistogram(handles)
temp=0;
stats=0;
plot(handles.axes_histogram, 0:255, temp); H2 = gca;
grid on; semilogy(0:255, temp); xlim([0 255]); ylim([0 max(histo)]); title(H2, ['Max: ' '0' ', Mean : ' '0'])
hold on, plot(H2, [240 240],[1 max(temp)],'r--'), hold off

function updateHistogram(handles)
% temp1=rot90(handles.activex1.GetRawData,1);
% figure(1), imhist(temp);
histo=handles.activex1.GetHistogram(0);
stats=handles.activex1.GetImageStat(0,1);
plot(handles.axes_histogram, 0:255, histo); H2 = gca;
grid on; semilogy(0:255, histo); xlim([0 255]); ylim([0 max(histo)]); title(H2, ['Max: ' num2str(stats(5))]) % ', Mean : ' num2str(floor(stats(1)))
hold on, plot(H2, [240 240],[1 max(histo)],'r--'), hold off
% The artifacts from the spectral reconstruction are originated from
% over-exposed data. So, don't make the histogram is bigger than 150.

% --- Executes on selection change in magnification.
function magnification_Callback(hObject, eventdata, handles)
val=get(hObject,'Value');
vect=[0 .1 .25 .5 1 2 3 4 5];
handles.activex1.Magnification=vect(val);
% --- Executes during object creation, after setting all properties.
function magnification_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in rotation.
function rotation_Callback(hObject, eventdata, handles)
val=get(hObject,'Value');
handles.activex1.Rotate=(val-1)*90;
% --- Executes during object creation, after setting all properties.
function rotation_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in flip.
function flip_Callback(hObject, eventdata, handles)
val=get(hObject,'Value');
handles.activex1.Flip=val-1;
% --- Executes during object creation, after setting all properties.
function flip_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in mono_init.
function mono_init_Callback(hObject, eventdata, handles)
if get(handles.mono_init,'Value')
    handles.mono = JYInit();
    load fittedmodel1
    handles.monofit=fittedmodel1;
    %monoc=1;
    JYSetWavelength(handles.mono, fittedmodel1(550));
    disp('Mono init done')
    guidata(hObject, handles);
%     set(handles.light_width,'String',num2str(handles.mono.GetCurrentSlitWidth(0)));
%     set(handles.slit_width,'String',num2str(handles.mono.GetCurrentSlitWidth(2)));
    set(handles.wavelength,'String','550');
else
    handles.mono.delete;
    disp('Mono un-init')
end

% --- Executes on button press in piezo_init.
function piezo_init_Callback(hObject, eventdata, handles)
if get(handles.piezo_init,'Value')
    init_piezo_fun(hObject,handles);
    disp('Done with piezo rs232 init')
else
    fclose(handles.serial);
    delete(handles.serial);
    disp('Done with piezo rs232 clear')
end

function init_piezo_fun(hObject,handles)
% Init peizo
% reset the com port:
instrreset

% Open com port:
% reset the com port:
instrreset
% Open com port:
handles.serial= serial('COM4'); % check with system monitor
% Set port parameters:
handles.serial.StopBits = 1;
handles.serial.Parity = 'none';
handles.serial.DataBits = 8;
handles.serial.BaudRate = 19200;
handles.serial.Terminator= 'CR';
handles.serial.FlowControl = 'software'; % Enable hardward handshaking; RTS/CTS pins
% Open port:
fopen(handles.serial);
% set to closed loop:
ch=0; var=1; fprintf(handles.serial, ['cloop,' num2str(ch) ',' num2str(var)]); pause(.1), 
ch=1; var=1; fprintf(handles.serial, ['cloop,' num2str(ch) ',' num2str(var)]);
% set to remote control:
fprintf(handles.serial, 'setk,0,1');
fprintf(handles.serial, 'setk,1,1');
fprintf(handles.serial, 'setk,2,1');
% fprintf(s, 'setk,0,0');
% fprintf(s, 'setk,1,0');
% fprintf(s, 'setk,2,0');
% set location:
ch=0; var=0; fprintf(handles.serial, ['set,' num2str(ch) ',' num2str(var)]); pause(.01)
ch=1; var=0; fprintf(handles.serial, ['set,' num2str(ch) ',' num2str(var)]); pause(.01)
ch=2; var=0; fprintf(handles.serial, ['set,' num2str(ch) ',' num2str(var)]); 
guidata(hObject,handles);

function move_wait(pos_horz,pos_vert,handles)
if isempty(handles.serial)
    disp('Init driver first')
end
% tic
ind=0;
fprintf(handles.serial, ['setall,' num2str(pos_horz) ',' num2str(pos_vert) ',0']); 
%pause(.01)
pause(1);
while 1
    fprintf(handles.serial, 'measure'); 
    %pause(.005);
    pause(1);
    str=fscanf(handles.serial); 
    %pause(.001);
    pause(1);
    indx=findstr(str,','); 
    comp=str2num(str(indx(1)+1:indx(2)-1)); pause(0)
    %if comp <= pos_horz+1 & comp >= pos_horz-1 | ind > 4
    if comp <= pos_horz+1 && comp >= pos_horz-1 || ind > 4    
        break;
    end
    %pause(.01);
    pause(.01);
    ind=ind+1;
end
% toc
ind;


% --- Executes on button press in acquire.
function acquire_Callback(hObject, eventdata, handles)
%val = get(handles.rotation,'Value');
%handles.activex1.Rotate=(val-1)*90; % by Min
%path = [pwd '\'];
%set(handles.directory,'String', path); % by Min
%histogram_Callback(hObject, eventdata, handles); % by Min
if get(handles.acquire,'Value')
    startcamera(handles);
else
    endcamera(handles);
end
%updateHistogram(handles);

function startcamera(handles)
handles.activex1.PacketSize=9000;
handles.activex1.SizeX=2048;
handles.activex1.SizeY=2048;
handles.activex1.Acquire=true;
handles.activex1.Display=true; pause(0);
handles.activex1.Magnification=0;
handles.activex1.ScrollBars=1;
pause(2); updateHistogram(handles);


function endcamera(handles)
handles.activex1.Acquire=false;
handles.activex1.Display=false;
handles.activex1.ScrollBars=0;

% --- Executes on button press in auto_exp.
function auto_exp_Callback(hObject, eventdata, handles)
autoexposure(hObject,handles);
% updatePreview(handles);

function autoexposure(hObject,handles)
cur_exp=str2double(get(handles.exp_global,'String'));
% setexp_man(cur_exp,handles); pause(cur_exp/1000 + .05);
% set(handles.exp_global,'String',num2str(cur_exp)), pause(cur_exp/1000+.1);
info=handles.activex1.GetImageStat(0,1); mx=info(5); me=info(1);

if mx < 10
    cur_exp=cur_exp*7;
elseif mx < 50
    cur_exp=cur_exp*3;
elseif mx < 100
    cur_exp=cur_exp*2;
elseif mx < 130
    cur_exp=cur_exp*1.5;
elseif mx < 160
    cur_exp=cur_exp*1.2;
elseif mx == 255
    cur_exp=cur_exp/3;
end
setexp_man(cur_exp,handles);

set(handles.exp_global,'String',num2str(cur_exp));
info=handles.activex1.GetImageStat(0,1); mx=info(5); me=info(1);
cur_exp=cur_exp*245/mx/1.05;
pause(1); % set two sec
setexp_man(cur_exp, handles);
pause(1); % set two sec
set(handles.exp_global,'String',num2str(cur_exp));
info=handles.activex1.GetImageStat(0,1); mx=info(5); me=info(1);
cur_exp=cur_exp*240/mx/1.05;
pause(1); % set two sec
setexp_man(cur_exp, handles);
pause(1); % set two sec
set(handles.exp_global,'String',num2str(cur_exp));
handles.base_exp=cur_exp;
guidata(hObject, handles);
disp('Done with auto exposure')
%pause(cur_exp/1000+.1);
pause(1); % set two sec
info=handles.activex1.GetImageStat(0,1); mx=info(5);
disp(['exposure: ' num2str(cur_exp) ' msec']);
disp(['Max: ' num2str(round(mx))]);
updateHistogram(handles);

% if (mx == 255)
%     disp(['running autoexposure again.']);
%     autoexposure(hObject,handles);
% end

% ActiveGigE
%  Stat [0] – Mean value  
%  Stat [1] – Standard deviation  
%  Stat [2] – Pixel count  
%  Stat [3] – Minimum  
%  Stat [4] – Maximum  
%  Stat [5] – Median  
%  Stat [6] – Skewness  
%  Stat [7] – Kurtosis  

function [hdr, shutters] = capturehdr(hObject, eventdata ,handles)
% read exposure setting
exp_stops=str2double(get(handles.exp_stops,'String'));
% read number of exposure setting
exp_num=str2double(get(handles.exp_num,'String'));

%keep the original base exposure
org_exp=str2double(get(handles.exp_global,'String'));

% make one step under (highlight is more important)!
org_exp  = org_exp/2;

% ldr = zeros(2024,2024,'single');

% capture next exposure pictures
for i=1:exp_num
    % read current camera exposure
    cur_exp = org_exp;

    % recalculate exposure 
    cur_exp = cur_exp*(2^((i-1)*exp_stops));

%     if (cur_exp < 70)
%         cur_exp = 70;
%     else
    if (cur_exp > 10000) % 10,000 is the maximum time
        cur_exp = 10000;
        break;
    end
    
    % record shutters
    shutters(i) = cur_exp;

    disp([num2str(i) '. exposure: ' num2str(cur_exp) ' msec']);
    set(handles.exp_global,'String',num2str(cur_exp));
    
    % set exposure
    setexp_man(cur_exp,handles);   
    pause(cur_exp/1000+1);
    
    % capture base-exposure
    %handles.activex1.Grab;
%     if (i>1)
%         oldldr = ldr;
%     end
    ldr(:,:,i) = fliplr(rot90(handles.activex1.GetRawData,1));
    pause(cur_exp/1000+5);
    
    info=handles.activex1.GetImageStat(0,1); mx=info(5);
    disp(['Camera Max: ' num2str(round(mx))]);    
    disp(['Actual Max: ' num2str(max(max(ldr(:,:,i))))]);
    
    trial = 0;
    % if there is any error, in receiving the data
    while (uint8(mx) - uint8(max(max(ldr(:,:,i)))) > 50)
        trial = trial + 1;
        if (trial > 5)
            disp(['******************************']);
            disp(['**** data transfer error *****']);
            disp(['******************************']);
            ldr = oldldr; % put the previous back
            break;
        end
        disp(['Retransfering image...']);
        ldr(:,:,i) = fliplr(rot90(handles.activex1.GetRawData,1));
        pause(cur_exp/1000+5);

        info=handles.activex1.GetImageStat(0,1); mx=info(5);
        disp(['Camera Max: ' num2str(round(mx))]);    
        disp(['Actual Max: ' num2str(max(max(ldr(:,:,i))))]);
    end
%    imwrite(ldr(:,:,i),['source_' datestr(clock,30) '_' num2str(i) '.png']);
end

% restore the original exposure
setexp_man(org_exp*2,handles);   
set(handles.exp_global,'String',num2str(org_exp*2));

% make it HDR
hdr = cassimakehdr(ldr, shutters);
clear ldr;
% normalize hdr to 255 for compatibility with previous but not make it
% integer
%----------------------------------
% remove peak noise & normalization
h = fspecial('disk',10);
hdrblur = imfilter(hdr,h,'replicate');
maxhdr = max(max(max(hdrblur)));     
hdr = hdr./max(max(max(maxhdr))).*255.;
hdr(hdr<0) = 0;
hdr(hdr>255) = 255;
%----------------------------------
updateHistogram(handles);


% --- Executes on button press in take_capture.
function take_capture_Callback(hObject, eventdata, handles) %#ok<*INUSD,*DEFNU>
% reset the coded aperture
disp '<Reset the translation of the coded aperture>'; % Since 2012-1-3 (SIGGRAPH3)
move_wait(0,0,handles); pause(0.5);
% capture
if get(handles.checkbox_hdron,'Value')
    %==========================================================
    % HDR part
    hdr = capturehdr(hObject, eventdata ,handles);
    %==========================================================
    prefix = [get(handles.edit_prefix,'String') '_'];
    filen=[prefix 'raw_' datestr(clock,30) '.hdr'];

    hdr = repmat(hdr,[1 1 3]); % 1d -> 3d
    if ~isempty('handles.pathname')
    %    imwrite(I,[handles.pathname filen]); % DUKE
        writehdr(hdr,[handles.pathname filen]); % Min
    else
    %    imwrite(I,filen); % DUKE
        writehdr(hdr,filen); % Min
    end
    clear hdr;
else
    prefix = [get(handles.edit_prefix,'String') '_'];
    filen=[prefix 'raw_' datestr(clock,30) '.png'];
    ldr = fliplr(rot90(handles.activex1.GetRawData,1));
    if ~isempty('handles.pathname')
        imwrite(ldr,[handles.pathname filen]);
    else
        imwrite(ldr,filen);
    end
end
function do_multi_capture(hObject, eventdata, handles)
%clear all;
% cal_gui
% Take multiframe capture:
% global Ax2 Ax0 shift exposure I_trans
shutters = 0;
exp_time=str2double(get(handles.exp_global,'String'));
setexp_man(exp_time, handles);
handles.base_exp=exp_time;
piezo_set_all_Callback(hObject, eventdata, handles);

exposure=handles.base_exp;
num_translations=str2double(get(handles.num_frames,'String'));
mask_pix_size=7.4; %str2double(get(handles.pixel_size,'String'));
disp(['Pixel size set to: ' num2str(mask_pix_size) 'um'])
% Shift based on one pixel shift:
shift=round(rand(2,num_translations)*21)*mask_pix_size;
% shift=  [  12     0     3     7     3     6    14     9     5     3    11     2
%     10     7    17    11    13    14    16     2    19    17    21     9 ];
% shift=shift*mask_pix_size;
shift(:,1)=[0;0];

%shift=[0,9,12,21,11,1,5,14,9,19,2,16,10,6,8,8,21,22,1,18,13,21,7,5,17,14,22,7,24,20,11,6,12,7,11,19,16,1,7,24,19,14,23,17,1,16,9,20,9,21,16,16,10,22,12,10,24,9,6,21,13,4,11,22;0,19,12,8,23,23,16,16,15,0,23,6,3,6,4,3,2,10,8,19,16,1,1,17,21,2,19,13,17,10,13,18,16,3,9,19,3,13,23,7,22,21,13,14,11,13,22,20,14,22,5,2,16,19,18,23,21,11,19,22,14,22,5,18];
%shift=shift*mask_pix_size;

% shift=shift(:,1:12);
I_trans=zeros(handles.activex1.SizeY,handles.activex1.SizeX,size(shift,2),'single');
tstart = tic();
for ind=1:size(shift,2)
    disp(['#' num2str(ind) '_x:' num2str(round(shift(1,ind)/mask_pix_size))...
        ',y:' num2str(round(shift(2,ind)/mask_pix_size))]);
    move_wait(shift(1,ind),shift(2,ind),handles); pause(exposure/1000*3+.1)
    if get(handles.checkbox_hdron,'Value')
        % hdr version
        [I_trans(:,:,ind) shutters] = capturehdr(hObject, eventdata ,handles); % by Min (HDR)
    else
        % previous ldr version
        I_trans(:,:,ind)=fliplr(rot90(handles.activex1.GetRawData,1)); % by Min (LDR)
    end
    pause(exposure*10^-6)
end
telapsed = toc(tstart);
shift=round(shift/mask_pix_size);
prefix = [get(handles.edit_prefix,'String') '_'];
filen=[prefix 'raw_' datestr(clock,30)];

% read values
shutters=str2double(get(handles.exp_global,'String'));

ex_time = handles.exp_global/1000;
ex_gain = handles.activex1.GainAbs;
ex_stops = str2num(get(handles.exp_stops,'String'));
ex_num = str2num(get(handles.exp_num,'String'));
total_time = round(telapsed);

disp(['Time elapsed: ' num2str(total_time) ' sec']);


if ~isempty('handles.pathname')
    save([handles.pathname filen],'I_trans','shift','mask_pix_size',...
        'ex_time', 'ex_gain', 'ex_stops', 'ex_num', 'total_time', 'shutters');
else
    save(filen,'I_trans','shift','mask_pix_size',...
        'ex_time', 'ex_gain', 'ex_stops', 'ex_num', 'total_time', 'shutters');
end
disp('Done with start translation')
move_wait(0,0,handles);


% --- Executes on button press in start_multi.
function start_multi_Callback(hObject, eventdata, handles)
do_multi_capture(hObject, eventdata, handles); % first run
do_multi_capture(hObject, eventdata, handles); % second run
msgbox('Done!','Message','warn');beep;


function piezo_set_all_Callback(hObject, eventdata, handles)
temp=str2double(get(hObject,'String'));
fprintf(handles.serial, ['setall,' num2str(temp) ',' num2str(temp) ',0']); 

% --- Executes during object creation, after setting all properties.
function piezo_set_all_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function wavelength_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function exp_global_Callback(hObject, eventdata, handles)
exp_time=str2double(get(handles.exp_global,'String'));
setexp_man(exp_time, handles);
handles.base_exp=exp_time;
guidata(hObject,handles);
pause(exp_time/1000+.05);
info=handles.activex1.GetImageStat(0,1); mx=info(5);
disp(['Max: ' num2str(floor(mx))]);
histogram_Callback(hObject, eventdata, handles);
% updatePreview(handles);

% --- Executes during object creation, after setting all properties.
function exp_global_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%=========================================================================
% exposure control is not clear!!!
% Exillary functions ------------------------------------------------------
% This function uses frameTime instead of fps, gives up to 10us resolution
% rather than 1fps resolution
function setexp_man(exp_time, handles)
handles.activex1.Acquire=false; pause(0.0);
% exp_time assumed to be in msec
% Long int: 120ms-10,000ms
% fps:      66.7ms-500ms
% timed:    .070ms-63.18ms
if exp_time > 500 % Long int, 0, 120min, 10inc, 10,000max
    exp_time=round(exp_time/10)*10; % set to inc of only 10
    long_int(exp_time,handles);    
elseif exp_time >= (1/.015) && exp_time <= 500  % Set fps, 0, 333min,10inc, 500000max
    exp_time=round(exp_time*100)*10;
    fps(exp_time,handles);
elseif exp_time <= (1/.015) % Timed % Timed, inc=10, 70=min
    exp_time=round(exp_time*100)*10;
    timed(exp_time,handles);
else
    disp('No exp set')    
end
handles.activex1.Acquire=true; pause(0.0);
updateHistogram(handles);


function long_int(exp_time,handles)
%disp long_int
handles.activex1.ExposureMode='Off'; pause(.01);
handles.activex1.AcquisitionFrameRateAbs=0; pause(.01);
handles.activex1.WriteRegister(hex2dec('E858'),exp_time);pause(5);

function fps(exp_time,handles)
%disp fps
handles.activex1.ExposureMode='Off'; pause(.01);
handles.activex1.WriteRegister(hex2dec('E858'),0); pause(.01);
handles.activex1.WriteRegister(hex2dec('E864'),exp_time); pause(1);
% handles.activex1.AcquisitionFrameRateAbs=round(1/(exp_time/1000));

function timed(exp_time,handles)
%disp timed
handles.activex1.WriteRegister(hex2dec('E858'),0); pause(.01);
handles.activex1.AcquisitionFrameRateAbs=0; pause(.01);
handles.activex1.ExposureMode='Timed'; pause(.01);
if exp_time > 63180
    exp_time=63170;
end
%handles.activex1.AcquisitionFrameRateAbs=15; pause(.01);
handles.activex1.ExposureTimeRaw=exp_time;pause(1);
%=========================================================================


function exp_gain_Callback(hObject, eventdata, handles)
gain=str2double(get(handles.exp_gain,'String'));
ex_max=handles.activex1.GetFeatureMax('GainAbs');
ex_min=handles.activex1.GetFeatureMin('GainAbs');
if gain <= ex_max && gain >= ex_min
    handles.activex1.GainAbs=gain;
else
    if gain<ex_max
        handles.activex1.GainAbs=ex_min;
        set(handles.gain,'String',num2str(ex_min));
    elseif gain > ex_min
        handles.activex1.GainAbs=ex_max;
        set(handles.gain,'String',num2str(ex_max));
    end
    disp('Gain out of range')
end
updateHistogram(handles);
% updatePreview(handles);

% --- Executes during object creation, after setting all properties.
function exp_gain_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in show_prop.
function show_prop_Callback(hObject, eventdata, handles)
handles.activex1.ShowProperties



function directory_Callback(hObject, eventdata, handles)
handles.pathname=[get(hObject,'String') '\'];
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function directory_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function pathname_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function slit_width_Callback(hObject, eventdata, handles)
slit_width=str2double(get(hObject,'String'));
if slit_width > 2.24
    slit_width=2.24;
elseif slit_width <0
    slit_width=0;
end
set(handles.slit_width,'String',num2str(slit_width));
% curr_width=JYChangeSlit(handles.mono, 0, slit_width);
curr_width=JYChangeSlit(handles.mono, 2, slit_width);
curr_width=JYChangeSlit(handles.mono, 0, slit_width);

% --- Executes during object creation, after setting all properties.
function slit_width_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in histogram.
function histogram_Callback(hObject, eventdata, handles)
updateHistogram(handles);
% 
% % temp1=rot90(handles.activex1.GetRawData,1);
% % figure(1), imhist(temp);
% temp=handles.activex1.GetHistogram(0);
% stats=handles.activex1.GetImageStat(0,1);
% figure(123), semilogy(0:255, temp); title(['Max: ' num2str(stats(5)) ', Mean : ' num2str(stats(1))])
% hold on, plot([255 255],[1 max(temp)],'r--'), hold off
% % figure, imagesc(temp1)



function exp_stops_Callback(hObject, eventdata, handles)
% hObject    handle to exp_stops (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of exp_stops as text
%        str2double(get(hObject,'String')) returns contents of exp_stops as a double


% --- Executes during object creation, after setting all properties.
function exp_stops_CreateFcn(hObject, eventdata, handles)
% hObject    handle to exp_stops (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function exp_num_Callback(hObject, eventdata, handles)
% hObject    handle to exp_num (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of exp_num as text
%        str2double(get(hObject,'String')) returns contents of exp_num as a double


% --- Executes during object creation, after setting all properties.
function exp_num_CreateFcn(hObject, eventdata, handles)
% hObject    handle to exp_num (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in turn_p45.
function turn_p45_Callback(hObject, eventdata, handles)
% hObject    handle to turn_p45 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
!mkturn +45

% --- Executes on button press in turn_n45.
function turn_n45_Callback(hObject, eventdata, handles)
% hObject    handle to turn_n45 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
!mkturn -45

% --- Executes on button press in turn_p30.
function turn_p30_Callback(hObject, eventdata, handles)
% hObject    handle to turn_p30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
!mkturn +30

% --- Executes on button press in turn_n30.
function turn_n30_Callback(hObject, eventdata, handles)
% hObject    handle to turn_n30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
!mkturn -30


% --- Executes on button press in low_laser.
function low_laser_Callback(hObject, eventdata, handles)
% hObject    handle to low_laser (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fn=[get(handles.edit_low_browse,'String')];
onescan('low',fn, handles);

% --- Executes on button press in high_laser.
function high_laser_Callback(hObject, eventdata, handles)
% hObject    handle to high_laser (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fn=[get(handles.edit_high_browse,'String')];
onescan('high',fn, handles);


% --- Executes on button press in push_high_browse.
function push_high_browse_Callback(hObject, eventdata, handles)
% hObject    handle to push_high_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% file read
[file_name,file_path] = uigetfile({'*.swd','swd';},'Choose a scanner profile');
if iscell(file_name)==0;
    if file_name==0; return; end;
end
source_file = strcat(file_path,file_name);    
filename = char(source_file);
set(handles.edit_high_browse,'String', filename); 


function edit_high_browse_Callback(hObject, eventdata, handles)
% hObject    handle to edit_high_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_high_browse as text
%        str2double(get(hObject,'String')) returns contents of edit_high_browse as a double


% --- Executes during object creation, after setting all properties.
function edit_high_browse_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_high_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_low_browse.
function push_low_browse_Callback(hObject, eventdata, handles)
% hObject    handle to push_low_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file_name,file_path] = uigetfile({'*.swd','swd';},'Choose a scanner profile');
if iscell(file_name)==0;
    if file_name==0; return; end;
end
source_file = strcat(file_path,file_name);    
filename = char(source_file);
set(handles.edit_low_browse,'String', filename); 



function edit_low_browse_Callback(hObject, eventdata, handles)
% hObject    handle to edit_low_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_low_browse as text
%        str2double(get(hObject,'String')) returns contents of edit_low_browse as a double


% --- Executes during object creation, after setting all properties.
function edit_low_browse_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_low_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_spin.
function pushbutton_spin_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_spin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
!mkturn 180


% --- Executes on button press in checkbox_hdron.
function checkbox_hdron_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_hdron (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_hdron
% if get(handles.checkbox_hdron,'Value')
%     % hdr version
%     set(exp_stops,
%     set(exp_num,
% else
%     set(exp_stops,
%     set(exp_num,
% end



function edit_prefix_Callback(hObject, eventdata, handles)
% hObject    handle to edit_prefix (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_prefix as text
%        str2double(get(hObject,'String')) returns contents of edit_prefix as a double
updateHistogram(handles);


% --- Executes during object creation, after setting all properties.
function edit_prefix_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_prefix (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
