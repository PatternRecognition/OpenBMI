function varargout = MI_Analy(varargin)
% MI_ANALY MATLAB code for MI_Analy.fig
%      MI_ANALY, by itself, creates a new MI_ANALY or raises the existing
%      singleton*.
%
%      H = MI_ANALY returns the handle to a new MI_ANALY or the handle to
%      the existing singleton*.
%
%      MI_ANALY('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MI_ANALY.M with the given input arguments.
%
%      MI_ANALY('Property','Value',...) creates a new MI_ANALY or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MI_Analy_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MI_Analy_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MI_Analy

% Last Modified by GUIDE v2.5 06-Sep-2017 16:28:37

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MI_Analy_OpeningFcn, ...
                   'gui_OutputFcn',  @MI_Analy_OutputFcn, ...
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


% --- Executes just before MI_Analy is made visible.
function MI_Analy_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MI_Analy (see VARARGIN)

% Choose default command line output for MI_Analy
handles.output = hObject;

axes(handles.axes1);
imshow('right.jpg');
axes(handles.axes2);
imshow('left.jpg');
axes(handles.axes3);
imshow('down.jpg');
% axes(handles.Graph);
% imshow('bar_demo7.png');
% axes(handles.Graph1);
% imshow('All3.png');


set(handles.Stimulus1,'Value',1);
% set(handles.Offline,'Value',1);
set(handles.Selection,'Visible','off');
% set(handles.OnlinePannel,'Visible','off');
% set(handles.Visualization,'Visible','off');
% set(handles.Analysispannel,'Visible','on');

set(handles.SamplingRate,'String',100);
set(handles.Band1,'String',8); set(handles.Band2,'String',30);
set(handles.Interval1,'String',0); set(handles.Interval2,'String',3500);



% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MI_Analy wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MI_Analy_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



% function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
% function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
% if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
%     set(hObject,'BackgroundColor','white');
% end


% --- Executes on button press in Load.
function Load_Callback(hObject, eventdata, handles)
% hObject    handle to Load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file,path]=uigetfile('*.eeg','Load EEG file (.eeg)');
file=strsplit(file,'.eeg');
fs=str2double(get(handles.SamplingRate,'String'));

% type of Que
if get(handles.Stimulus1,'Value')
    marker1={'1','right'};
else
    marker1={};
end

if get(handles.Stimulus2,'Value')
    marker2={'2','left'};
else
    marker2={};
end

if get(handles.Stimulus3,'Value')
    marker3={'3','foot'};
else
    marker3={};
end
marker=[marker1;marker2;marker3];

[EEG.data, EEG.marker, EEG.info]=Load_EEG([path,file{1}],{'device','brainVision';'marker', marker;'fs',fs});
handles.EEG=EEG;
dataSize=size(EEG.data.x);
% description0=sprintf('Subject:');
description1=sprintf('Data size: %.0fx%.0f',dataSize(1),dataSize(2));
description2=sprintf('Number of channel: %d',dataSize(2));
description3=sprintf('Number of class: %d',EEG.marker.nClasses);
description4=sprintf('Number of trial: %d',length(EEG.marker.y_class));
description5=sprintf('Original sampling rate: %d',EEG.info.orig_fs);
% description6=sprintf('right: 50\nleft: 50\nfoot: 50');

description={description1;description5;description2;...
             description3;description4};
% description={description0;description1;description5;description2;...
%              description3;description4;description6};

set(handles.Description,'HorizontalAlignment','left');
set(handles.Description,'String',description);

guidata(hObject, handles);


% --- Executes on button press in REMOVE.
% function REMOVE_Callback(hObject, eventdata, handles)
% % hObject    handle to REMOVE (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)



% function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
% function edit2_CreateFcn(hObject, eventdata, handles)
% % hObject    handle to edit2 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    empty - handles not created until after all CreateFcns called
% 
% % Hint: edit controls usually have a white background on Windows.
% %       See ISPC and COMPUTER.
% if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
%     set(hObject,'BackgroundColor','white');
% end



% function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
% function edit3_CreateFcn(hObject, eventdata, handles)
% % hObject    handle to edit3 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    empty - handles not created until after all CreateFcns called
% 
% % Hint: edit controls usually have a white background on Windows.
% %       See ISPC and COMPUTER.
% if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
%     set(hObject,'BackgroundColor','white');
% end


% --- Executes on button press in pushbutton3.
% function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in popupmenu1.
% function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton1.
% function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1



function Band2_Callback(hObject, eventdata, handles)
% hObject    handle to Band2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Band2 as text
%        str2double(get(hObject,'String')) returns contents of Band2 as a double


% --- Executes during object creation, after setting all properties.
function Band2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Band2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Band1_Callback(hObject, eventdata, handles)
% hObject    handle to Band1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Band1 as text
%        str2double(get(hObject,'String')) returns contents of Band1 as a double


% --- Executes during object creation, after setting all properties.
function Band1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Band1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3


% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Interval2_Callback(hObject, eventdata, handles)
% hObject    handle to Interval2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Interval2 as text
%        str2double(get(hObject,'String')) returns contents of Interval2 as a double


% --- Executes during object creation, after setting all properties.
function Interval2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Interval2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Interval1_Callback(hObject, eventdata, handles)
% hObject    handle to Interval1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Interval1 as text
%        str2double(get(hObject,'String')) returns contents of Interval1 as a double


% --- Executes during object creation, after setting all properties.
function Interval1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Interval1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton5.
% function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in radiobutton2.
% function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2


% --- Executes on button press in pushbutton4.
% function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in Connection.
function Connection_Callback(hObject, eventdata, handles)
% hObject    handle to Connection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
EEG=handles.EEG;
band1=str2double(get(handles.Band1,'String'));
band2=str2double(get(handles.Band2,'String'));
band=[band1 band2];

Interval1=str2double(get(handles.Interval1,'String'));
Interval2=str2double(get(handles.Interval2,'String'));
Interval=[Interval1 Interval2];
t_stimulus=Interval2-Interval1;

ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    nClass=1;
elseif strcmp(temp1,'Two-class')
    nClass=2;
else
    nClass=3;
end

fs=str2double(get(handles.SamplingRate,'String'));

% Channel_index  채널 셀렉션 부분 해줘야함. 일단은....
channel_index=1:32; % (FC1~6,C1~6,Cz,CP1~6)
% channel_index=[12, 13, 14,     57, 56, 55 ...
%                17, 18, 19, 20, 52, 51, 50 ...
%                23, 24, 25,     48, 47, 46]; % (FC1~6,C1~6,Cz,CP1~6) 19개
% 클래스 1개일때, 2개일때, 3개일때... 어떤 클래스 선택됫는지 그거 해줘야됨
[LOSS, CSP, LDA]=MI_calibration_new(EEG, band, fs, Interval, {'nClass',nClass;'channel',channel_index});

Feedback_Client_new(CSP, LDA, band, fs, t_stimulus,handles.Graph, handles.Graph1,{'buffer_size',5000; 'data_size',1000; 'channel',channel_index; 'feedback_freq',100/1000; 'TCPIP','on'});



% --- Executes on button press in Reset.
% function Reset_Callback(hObject, eventdata, handles)
% hObject    handle to Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in Stimulus1.
function Stimulus1_Callback(hObject, eventdata, handles)
% hObject    handle to Stimulus1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    if (get(handles.Stimulus1,'Value'))
        set(handles.Stimulus2,'Value',0)
        set(handles.Stimulus3,'Value',0)
    else
        if (get(handles.Stimulus2,'Value'))
            set(handles.Stimulus3,'Value',0)
        else
            set(handles.Stimulus3,'Value',1)
        end
    end
elseif strcmp(temp1,'Three-class')
    set(handles.Stimulus1,'Value',1)
else
    if (get(handles.Stimulus1,'Value'))
        if (get(handles.Stimulus2,'Value'))
            set(handles.Stimulus3,'Value',0)
        else
            set(handles.Stimulus3,'Value',1)
        end
    else
        set(handles.Stimulus2,'Value',1)
        set(handles.Stimulus3,'Value',1)
    end
end
% Hint: get(hObject,'Value') returns toggle state of Stimulus1


% --- Executes on button press in Stimulus2.
function Stimulus2_Callback(hObject, eventdata, handles)
% hObject    handle to Stimulus2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    if (get(handles.Stimulus2,'Value'))
        set(handles.Stimulus1,'Value',0)
        set(handles.Stimulus3,'Value',0)
    else
        if (get(handles.Stimulus3,'Value'))
            set(handles.Stimulus1,'Value',0)
        else
            set(handles.Stimulus1,'Value',1)
        end
    end
elseif strcmp(temp1,'Three-class')
    set(handles.Stimulus2,'Value',1)
else
    if (get(handles.Stimulus2,'Value'))
        if (get(handles.Stimulus3,'Value'))
            set(handles.Stimulus1,'Value',0)
        else
            set(handles.Stimulus1,'Value',1)
        end
    else
        set(handles.Stimulus1,'Value',1)
        set(handles.Stimulus3,'Value',1)
    end
end
% Hint: get(hObject,'Value') returns toggle state of Stimulus2


% --- Executes on button press in Stimulus3.
function Stimulus3_Callback(hObject, eventdata, handles)
% hObject    handle to Stimulus3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    if (get(handles.Stimulus3,'Value'))
        set(handles.Stimulus1,'Value',0)
        set(handles.Stimulus2,'Value',0)
    else
        if (get(handles.Stimulus1,'Value'))
            set(handles.Stimulus2,'Value',0)
        else
            set(handles.Stimulus2,'Value',1)
        end
    end
elseif strcmp(temp1,'Three-class')
    set(handles.Stimulus3,'Value',1)
else
    if (get(handles.Stimulus3,'Value'))
        if (get(handles.Stimulus1,'Value'))
            set(handles.Stimulus2,'Value',0)
        else
            set(handles.Stimulus2,'Value',1)
        end
    else
        set(handles.Stimulus1,'Value',1)
        set(handles.Stimulus2,'Value',1)
    end
end
% Hint: get(hObject,'Value') returns toggle state of Stimulus3


% --- Executes on selection change in NumofClass.
function NumofClass_Callback(hObject, eventdata, handles)
% hObject    handle to NumofClass (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    set(handles.Stimulus1,'Value',1);
    set(handles.Stimulus2,'Value',0);
    set(handles.Stimulus3,'Value',0);
elseif strcmp(temp1,'Two-class')
    set(handles.Stimulus1,'Value',1);
    set(handles.Stimulus2,'Value',1);
    set(handles.Stimulus3,'Value',0);
else
    set(handles.Stimulus1,'Value',1);
    set(handles.Stimulus2,'Value',1);
    set(handles.Stimulus3,'Value',1);
end
% Hints: contents = cellstr(get(hObject,'String')) returns NumofClass contents as cell array
%        contents{get(hObject,'Value')} returns selected item from NumofClass


% --- Executes during object creation, after setting all properties.
function NumofClass_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NumofClass (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
% function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
% if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
%     set(hObject,'BackgroundColor','white');
% end


% --- Executes on button press in Offline.
% function Offline_Callback(hObject, eventdata, handles)
% % hObject    handle to Offline (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% if (get(handles.Offline,'Value'))
%     set(handles.Online,'Value',0)
% %     set(handles.Visualization,'Visible','off');
% %     set(handles.Analysispannel,'Visible','on');
% %     set(handles.OnlinePannel,'Visible','off');
% %     set(handles.OfflinePannel,'Visible','on');
% %     set(handles.uipanel4,'visible','off')
% %     set(handles.uipanel5,'visible','off')
% else
%     set(handles.Online,'Value',1)
% %     set(handles.Visualization,'Visible','on');
% %     set(handles.Analysispannel,'Visible','off');
% %     set(handles.OnlinePannel,'Visible','on');
% %     set(handles.OfflinePannel,'Visible','off');
% %     set(handles.uipanel4,'visible','on')
% %     set(handles.uipanel5,'visible','on')
% end
% Hint: get(hObject,'Value') returns toggle state of Offline


% --- Executes on button press in Online.
% function Online_Callback(hObject, eventdata, handles)
% % hObject    handle to Online (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% if (get(handles.Online,'Value'))
%     set(handles.Offline,'Value',0)
% %     set(handles.Visualization,'Visible','on');
% %     set(handles.Analysispannel,'Visible','off');
% %     set(handles.OnlinePannel,'Visible','on');
% %     set(handles.OfflinePannel,'Visible','off');
% %     set(handles.uipanel4,'visible','on')
% %     set(handles.uipanel5,'visible','on')
% else
%     set(handles.Offline,'Value',1)
% %     set(handles.Visualization,'Visible','off');
% %     set(handles.Analysispannel,'Visible','on');
% %     set(handles.OnlinePannel,'Visible','off');
% %     set(handles.OfflinePannel,'Visible','on');
% %     set(handles.uipanel4,'visible','off')
% %     set(handles.uipanel5,'visible','off')
% end
% Hint: get(hObject,'Value') returns toggle state of Online


% --- Executes on selection change in Channel.
function Channel_Callback(hObject, eventdata, handles)
% hObject    handle to Channel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ncls=cellstr(get(handles.Channel,'String'));
temp1=ncls{get(handles.Channel,'Value')};
if strcmp(temp1,'All Channels')
    set(handles.Selection,'Visible','off');
else
    set(handles.Selection,'Visible','on');
end

% Hints: contents = cellstr(get(hObject,'String')) returns Channel contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Channel


% --- Executes during object creation, after setting all properties.
function Channel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Channel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Selection.
function Selection_Callback(hObject, eventdata, handles)
% hObject    handle to Selection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% [chan, p, o] = filt_EEG_CHANNEL(handles.EEG.info.chan);
% select_channel_v2({'chan',chan(2,:)});
handles.oout=select_channel_v2({'chan',handles.EEG.info.chan});
% ch_idx=oout.UserData;
% handles.ch_idx=oout.UserData;
guidata(hObject, handles);




function SamplingRate_Callback(hObject, eventdata, handles)
% hObject    handle to SamplingRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of SamplingRate as text
%        str2double(get(hObject,'String')) returns contents of SamplingRate as a double


% --- Executes during object creation, after setting all properties.
function SamplingRate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SamplingRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu6.
function popupmenu6_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu6 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu6


% --- Executes during object creation, after setting all properties.
function popupmenu6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function accur_Callback(hObject, eventdata, handles)
% hObject    handle to accur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of accur as text
%        str2double(get(hObject,'String')) returns contents of accur as a double


% --- Executes during object creation, after setting all properties.
function accur_CreateFcn(hObject, eventdata, handles)
% hObject    handle to accur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Accuracy.
function Accuracy_Callback(hObject, eventdata, handles)
% hObject    handle to Accuracy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
EEG=handles.EEG;
band1=str2double(get(handles.Band1,'String'));
band2=str2double(get(handles.Band2,'String'));
band=[band1 band2];

Interval1=str2double(get(handles.Interval1,'String'));
Interval2=str2double(get(handles.Interval2,'String'));
Interval=[Interval1 Interval2];

ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    nClass=1;
elseif strcmp(temp1,'Two-class')
    nClass=2;
else
    nClass=3;
end

fs=str2double(get(handles.SamplingRate,'String'));

% Channel_index  채널 셀렉션 부분 해줘야함. 일단은....
channel_index=1:32; % (FC1~6,C1~6,Cz,CP1~6)
% channel_index=[12, 13, 14,     57, 56, 55 ...
%                17, 18, 19, 20, 52, 51, 50 ...
%                23, 24, 25,     48, 47, 46]; % (FC1~6,C1~6,Cz,CP1~6) 19개
% 클래스 1개일때, 2개일때, 3개일때... 어떤 클래스 선택됫는지 그거 해줘야됨
[LOSS, CSP, LDA]=MI_calibration_yj(EEG, band, fs, Interval, {'nClass',nClass;'channel',channel_index});
acc=(1-LOSS{1,1})*100;
set(handles.accur,'String',acc);



% --- Executes on button press in pushbutton11.
% function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton12.
% function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton13.
% function pushbutton13_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton14.
% function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
