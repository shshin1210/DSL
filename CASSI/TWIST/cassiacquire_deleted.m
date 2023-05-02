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
% function exp_curve_Callback(hObject, eventdata, handles)
% --- Executes on button press in exp_curve.
function exp_curve_Callback(hObject, eventdata, handles)
wl=425:25:800;
% temp=zeros(handles.activex1.SizeY,handles.activex1.SizeX,length(wl),'uint8');
mx=zeros(length(wl),1); me=zeros(length(wl),1);
for ind=1:length(wl)
    wlout=JYSetWavelength(handles.mono, wl(ind)); pause(handles.base_exp/1000*2)
    %temp(:,:,ind)=rot90(handles.activex1.GetRawData,1); pause(.1);
    info=handles.activex1.GetImageStat(0,1);
    mx(ind)=info(5); me(ind)=info(1);
    set(handles.wavelength,'String',num2str(wlout,3));
end
handles.exp_factor=handles.base_exp*170./mx/1.05;
% plot(handles.axes_histogram,wl,mx,wl,me)
guidata(hObject,handles);
disp('Done with exp curve')

% --- Executes on button press in exp_check.
function exp_check_Callback(hObject, eventdata, handles)
wl=425:25:800;
% temp=zeros(handles.activex1.SizeY,handles.activex1.SizeX,length(wl),'uint8');
mx=zeros(length(wl),1); me=zeros(length(wl),1);
for ind=1:length(wl)
    setexp_man(handles.exp_factor(ind), handles);
    wlout=JYSetWavelength(handles.mono, wl(ind)); 
    pause(handles.exp_factor(ind)/1000*2+.3);
    set(handles.wavelength,'String',num2str(wlout,3));
    %temp(:,:,ind)=rot90(handles.activex1.GetRawData,1); pause(.1);
    info=handles.activex1.GetImageStat(0,1);
    mx(ind)=info(5); me(ind)=info(1);
    while mx(ind) == 255
        handles.exp_factor(ind)=handles.exp_factor(ind)*.7;
        setexp_man(handles.exp_factor(ind), handles); pause(handles.exp_factor(ind)/1000+.1);
        info=handles.activex1.GetImageStat(0,1);
        mx(ind)=info(5); me(ind)=info(1);
        disp('Over-corrected')
    end
end
handles.exp_factor=handles.exp_factor*230./mx/1.05;
% handles.exp_factor=200./handles.exp_factor;
% Fit to model:
ft_ = fittype('pchipinterp');
% Fit this model using new data
handles.exp_curve = fit(wl',handles.exp_factor,ft_);
% plot(handles.axes_histogram,wl,mx,wl,me)
guidata(hObject,handles);


% --- Executes on button press in disp_cuve.
function disp_cuve_Callback(hObject, eventdata, handles)
wl=425:25:800;
temp=zeros(handles.activex1.SizeY,handles.activex1.SizeX,length(wl),'uint8');
for ind=1:length(wl)
    setexp_man(handles.exp_factor(ind), handles);
    wlout=JYSetWavelength(handles.mono, wl(ind));
    set(handles.wavelength,'String',num2str(wlout,3));
    pause(handles.exp_factor(ind)/1000*2+.3);
    temp(:,:,ind)=fliplr(rot90(handles.activex1.GetRawData,1));
end
shift=zeros(2,length(wl));
for ind=2:size(temp,3)
    [output Greg] = dftregistration(fft2(temp(:,:,1)),fft2(temp(:,:,ind)),1);
    shift(:,ind)=output(3:4);
end
% plot(handles.axes_histogram,wl,shift)
handles.shift=shift;
% Fit to model:
% Fit this model using new data
st_ = [19.005565759825043 0.0012617502983467116 -1006.4101864617297 -0.0085934005392091679 ];
ft_ = fittype('exp2');
% Fit this model using new data
handles.dispersion_curve = fit(wl',handles.shift(2,:)',ft_,'Startpoint',st_);
guidata(hObject,handles);


% --- Executes on button press in calcube.
function calcube_Callback(hObject, eventdata, handles)
% First, find correct exposures, using exp_factor as baseline:
ft_ = fittype('pchipinterp');
% Fit this model using new data
wl=linspace(425,800,200);
x=handles.dispersion_curve(wl);
wvls_fit = fit(x,wl',ft_);
mono_input=wvls_fit(round(handles.dispersion_curve(425)):1:round(handles.dispersion_curve(800)));
exp_curve=handles.exp_curve(mono_input);
save('calube','mono_input','exp_curve','wvls_fit');
calcube=zeros(handles.activex1.SizeY,handles.activex1.SizeX,length(mono_input),'uint8');
mx=zeros(length(mono_input),1);
for ind=1:length(mono_input)
    setexp_man(exp_curve(ind), handles);
    wlout=JYSetWavelength(handles.mono, mono_input(ind));
    set(handles.wavelength,'String',num2str(wlout,3)); 
    pause(exp_curve(ind)/1000*2+.1);
    calcube(:,:,ind)=fliplr(rot90(handles.activex1.GetRawData,1));
    mx(ind) = max(reshape(calcube(:,:,ind),size(calcube,1)*size(calcube,2),1));
    if mx(ind)==255
        disp('Warning, clipped')
    end
end
% plot(handles.axes_histogram,mono_input,mx)filen=['raw_' datestr(clock,30)];
disp_locations=handles.shift;
save([handles.pathname 'calcube'],'mono_input','exp_curve','wvls_fit','calcube',...
    'mx','disp_locations');
disp('Done with calcube')


% --- Executes on button press in size_mask.
function size_mask_Callback(hObject, eventdata, handles)
while get(hObject,'Value')
    % Check for magnification:
    temp=fliplr(rot90(handles.activex1.GetRawData,1));
    I1=temp(1000:1050,:); tt=I1(:); tt=tt.*uint8(tt>20); left=floor(find(tt,1)/size(I1,1));
    tt=flipdim(I1,2); tt=tt(:); tt=tt.*uint8(tt>20); right=size(I1,2)-floor(find(tt,1)/size(I1,1));
    set(handles.txt_size_mask,'String',['Size: ' num2str(right-left)]);
    pause(.05);
    disp('done')
end

% --- Executes on button press in ref_taken.
function ref_taken_Callback(hObject, eventdata, handles)
global ref_check
if get(hObject,'Value')
    ref_check=fliplr(rot90(handles.activex1.GetRawData,1));
    disp('Reference set')
else
    ref_check=[];
end


% --- Executes on button press in motion_check.
function motion_check_Callback(hObject, eventdata, handles)
global ref_check
if ~isempty(ref_check)
    temp=fliplr(rot90(handles.activex1.GetRawData,1));
    [output Greg] = dftregistration(fft2(ref_check),fft2(temp),1);
    set(handles.ref_check,'String',num2str([output(4) output(3)]));
else
    disp('No reference set yet')
end



function wavelength_Callback(hObject, eventdata, handles)
wl=str2double(get(hObject,'String'));
%wlout=JYSetWavelength(handles.mono, handles.monofit(wl));
wlout=JYSetWavelength(handles.mono, wl);
disp(['Wavelength: ' num2str(wlout) ' nm']);

% if get(handles.mono_check,'Value')
%     % Check image if using monochrometer:
%     %montage2(data)
%     I=zeros(handles.activex1.SizeY,handles.activex1.SizeX);
%     data=double(I_trans);
%     for ind=1:size(shift,2)
%         if mod(ind,2)
%             I=I+circshift(data(:,:,ind),[-shift(2,ind),shift(1,ind)]);
%         else
%             I=I-circshift(data(:,:,ind),[-shift(2,ind),shift(1,ind)]);
%         end
%     end
%     figure, imshow(I,[])
% end

