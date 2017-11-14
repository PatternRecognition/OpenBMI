function varargout = MI_Para(varargin)
% MI_PARA MATLAB code for MI_Para.fig
%      MI_PARA, by itself, creates a new MI_PARA or raises the existing
%      singleton*.
%
%      H = MI_PARA returns the handle to a new MI_PARA or the handle to
%      the existing singleton*.
%
%      MI_PARA('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MI_PARA.M with the given input arguments.
%
%      MI_PARA('Property','Value',...) creates a new MI_PARA or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MI_Para_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MI_Para_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MI_Para

% Last Modified by GUIDE v2.5 04-Sep-2017 13:36:14

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MI_Para_OpeningFcn, ...
                   'gui_OutputFcn',  @MI_Para_OutputFcn, ...
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


% --- Executes just before MI_Para is made visible.
function MI_Para_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MI_Para (see VARARGIN)

% Choose default command line output for MI_Para
handles.output = hObject;

%
axes(handles.axes1);
imshow('right.jpg');
axes(handles.axes2);
imshow('left.jpg');
axes(handles.axes3);
imshow('down.jpg');


% screen size
set(handles.size1,'visible','off'); set(handles.size2,'visible','off');
set(handles.produc,'visible','off');
% set(handles.size1,'string','1920'); set(handles.size2,'string','1000');
% set(handles.size1,'Enable','off'); set(handles.size2,'Enable','off');

% Param
set(handles.TriggerPort,'string','C010');
set(handles.ScrNum,'string','2');
set(handles.NumofTrial,'string','50');
set(handles.Time_Stimulus,'string','3');
set(handles.Time_Interval,'string','2');
set(handles.Time_Rest,'string','2');

set(handles.Stimulus1,'Value',1);
set(handles.Offline,'Value',1);


% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MI_Para wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MI_Para_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



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


% ncls=cellstr(get(handles.NumofClass,'String'));
% temp1=ncls{get(handles.NumofClass,'Value')};
% if strcmp(temp1,'One-class')
%     N_class=1;
% elseif strcmp(temp1,'Two-class')
%     N_class=2;
% else
%     N_class=3;
% end
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


% --- Executes on button press in Offline.
function Offline_Callback(hObject, eventdata, handles)
% hObject    handle to Offline (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (get(handles.Offline,'Value'))
    set(handles.Online,'Value',0)
else
    set(handles.Online,'Value',1)
end

% Hint: get(hObject,'Value') returns toggle state of Offline


% --- Executes on button press in Online.
function Online_Callback(hObject, eventdata, handles)
% hObject    handle to Online (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (get(handles.Online,'Value'))
    set(handles.Offline,'Value',0)
else
    set(handles.Offline,'Value',1)
end
% Hint: get(hObject,'Value') returns toggle state of Online



function Time_Stimulus_Callback(hObject, eventdata, handles)
% hObject    handle to Time_Stimulus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Time_Stimulus as text
%        str2double(get(hObject,'String')) returns contents of Time_Stimulus as a double


% --- Executes during object creation, after setting all properties.
function Time_Stimulus_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Time_Stimulus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Time_Interval_Callback(hObject, eventdata, handles)
% hObject    handle to Time_Interval (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Time_Interval as text
%        str2double(get(hObject,'String')) returns contents of Time_Interval as a double


% --- Executes during object creation, after setting all properties.
function Time_Interval_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Time_Interval (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Time_Rest_Callback(hObject, eventdata, handles)
% hObject    handle to Time_Rest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Time_Rest as text
%        str2double(get(hObject,'String')) returns contents of Time_Rest as a double


% --- Executes during object creation, after setting all properties.
function Time_Rest_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Time_Rest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function NumofTrial_Callback(hObject, eventdata, handles)
% hObject    handle to NumofTrial (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NumofTrial as text
%        str2double(get(hObject,'String')) returns contents of NumofTrial as a double


% --- Executes during object creation, after setting all properties.
function NumofTrial_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NumofTrial (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ScrNum_Callback(hObject, eventdata, handles)
% hObject    handle to ScrNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ScrNum as text
%        str2double(get(hObject,'String')) returns contents of ScrNum as a double


% --- Executes during object creation, after setting all properties.
function ScrNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ScrNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in screensize.
function screensize_Callback(hObject, eventdata, handles)
% hObject    handle to screensize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
contents=cellstr(get(handles.screensize,'String'));
scrtype=contents{get(handles.screensize,'Value')};
if strcmp(scrtype,'Full screen')
    set(handles.size1,'visible','off');set(handles.size2,'visible','off');set(handles.produc,'visible','off');
%     set(handles.size1,'string','1920');set(handles.size2,'string','1000');
%     set(handles.size1,'Enable','off');set(handles.size2,'Enable','off')
elseif strcmp(scrtype,'1920 x 1200')
    set(handles.size1,'visible','on');set(handles.size2,'visible','on');set(handles.produc,'visible','on')
    set(handles.size1,'string','1920');set(handles.size2,'string','1200');
    set(handles.size1,'Enable','off');set(handles.size2,'Enable','off')
elseif strcmp(scrtype,'1920 x 1080')
    set(handles.size1,'visible','on');set(handles.size2,'visible','on');set(handles.produc,'visible','on')
    set(handles.size1,'string','1920');set(handles.size2,'string','1080');
    set(handles.size1,'Enable','off');set(handles.size2,'Enable','off')
elseif strcmp(scrtype,'1280 x 1024')
    set(handles.size1,'visible','on');set(handles.size2,'visible','on');set(handles.produc,'visible','on')
    set(handles.size1,'string','1280');set(handles.size2,'string','1024');
    set(handles.size1,'Enable','off');set(handles.size2,'Enable','off')
elseif strcmp(scrtype,'800 x 600')
    set(handles.size1,'visible','on');set(handles.size2,'visible','on');set(handles.produc,'visible','on')
    set(handles.size1,'string','800');set(handles.size2,'string','600');
    set(handles.size1,'Enable','off');set(handles.size2,'Enable','off')
else
    set(handles.size1,'visible','on');set(handles.size2,'visible','on');set(handles.produc,'visible','on')
    set(handles.size1,'Enable','on');set(handles.size2,'Enable','on')
    set(handles.size1,'string',[]);set(handles.size2,'string',[]);
end

% Hints: contents = cellstr(get(hObject,'String')) returns screensize contents as cell array
%        contents{get(hObject,'Value')} returns selected item from screensize


% --- Executes during object creation, after setting all properties.
function screensize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to screensize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function size2_Callback(hObject, eventdata, handles)
% hObject    handle to size2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of size2 as text
%        str2double(get(hObject,'String')) returns contents of size2 as a double


% --- Executes during object creation, after setting all properties.
function size2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to size2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in startparadigm_off.
function startparadigm_off_Callback(hObject, eventdata, handles)
% hObject    handle to startparadigm_off (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
t_stimulus = str2double(get(handles.Time_Stimulus , 'String'));
t_isi=  str2double(get(handles.Time_Interval , 'String'));
t_rest=  str2double(get(handles.Time_Rest , 'String'));
N_screen=  str2double(get(handles.ScrNum , 'String'));
N_trial=  str2double(get(handles.NumofTrial , 'String'));
Port= get(handles.TriggerPort , 'String');

% screen size
hori=  str2double(get(handles.size1, 'String'));
verti=  str2double(get(handles.size2, 'String'));
size_scr=cellstr(get(handles.screensize,'String'));
size_scr=size_scr{get(handles.screensize,'Value')};
if strcmp(size_scr,'Full screen')
    Size_screen='full';
else
    Size_screen=[hori,verti];
end
% # of class and type of stimulus
ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    N_class=1;
elseif strcmp(temp1,'Two-class')
    N_class=2;
else
    N_class=3;
end
% type of stimulus
sti1=get(handles.Stimulus1,'Value');
sti2=get(handles.Stimulus2,'Value');
sti3=get(handles.Stimulus3,'Value');
typ_stim=[sti1,sti2,sti3];

% paradigm start _off
if get(handles.Offline,'Value')
Makeparadigm_MI_new({'time_sti',t_stimulus;'time_cross',t_isi;'time_blank',t_rest;...
    'num_trial',N_trial;'num_class',N_class;'type_sti',typ_stim; 'port',Port;...
    'time_jitter',0.1;'num_screen',N_screen;'size_screen',Size_screen});
end
% paradigm start _on
if get(handles.Online,'Value')
Makeparadigm_MI_feedback_new({'time_sti',t_stimulus;'time_cross',t_isi;'time_blank',t_rest;...
    'num_trial',N_trial;'num_class',N_class;'type_sti',typ_stim;'port',Port;...
    'time_jitter',0.1;'num_screen',N_screen;'size_screen',Size_screen});
end

function size1_Callback(hObject, eventdata, handles)
% hObject    handle to size1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of size1 as text
%        str2double(get(hObject,'String')) returns contents of size1 as a double


% --- Executes during object creation, after setting all properties.
function size1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to size1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in Triggercheck.
function Triggercheck_Callback(hObject, eventdata, handles)
% hObject    handle to Triggercheck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% % % % 전반적으로 좀 fancy하게 다듬을 필요가 이따
bbci_acquire_bv('close');
port=get(handles.TriggerPort,'String');
global IO_ADDR IO_LIB;
IO_ADDR=hex2dec(port);
IO_LIB=which('inpoutx64.dll');

% bbci_acquire_bv();
params = struct;
state = bbci_acquire_bv('init', params);
buffer_size=5000;
data_size=5000;
feedback_t=100/1000; % feedback frequency
Dat=zeros(buffer_size, size(1:32,2)); % 1:32 채널 인덱스 이거...
% 필요한 트리거 개 수 만큼 날려주고 다 제대로 날아가는지 확인...하는 방식인데...
% 자극의 개수를 받아오고
% waitSecs을 0.5 로 해놧는데... 요것이....
qwe=0;
ppWrite(IO_ADDR,111);WaitSecs(0.5); % 시작 트리거 같고
[data, markertime, markerdescr, state] = bbci_acquire_bv(state);
if markerdescr==111
    qwe=qwe+1;
end

ppWrite(IO_ADDR,5);WaitSecs(0.5); % 중간에 +?beep? 에 대한 트리거 같고
[data, markertime, markerdescr, state] = bbci_acquire_bv(state);
if markerdescr==5
    qwe=qwe+1;
end

ppWrite(IO_ADDR,222);WaitSecs(0.5); % 끝날때 트리거 같고
[data, markertime, markerdescr, state] = bbci_acquire_bv(state);
if markerdescr==222
    qwe=qwe+1;
end

% # of class and type of stimulus
ncls=cellstr(get(handles.NumofClass,'String'));
temp1=ncls{get(handles.NumofClass,'Value')};
if strcmp(temp1,'One-class')
    N_class=1;
elseif strcmp(temp1,'Two-class')
    N_class=2;
else
    N_class=3;
end

% 필요한 자극의 개수
sti1=get(handles.Stimulus1,'Value');
sti2=get(handles.Stimulus2,'Value');
sti3=get(handles.Stimulus3,'Value');
if sti1
    ppWrite(IO_ADDR,1);WaitSecs(0.5);
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    if markerdescr==1
        qwe=qwe+1;
    end
end
if sti2
    ppWrite(IO_ADDR,2);;WaitSecs(0.5);
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    if markerdescr==2
        qwe=qwe+1;
    end
%     [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
end
if sti3
    ppWrite(IO_ADDR,3);;WaitSecs(0.5);
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    if markerdescr==3
        qwe=qwe+1;
    end
end
% bbci_acquire_bv('close');


function TriggerPort_Callback(hObject, eventdata, handles)
% hObject    handle to TriggerPort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TriggerPort as text
%        str2double(get(hObject,'String')) returns contents of TriggerPort as a double


% --- Executes during object creation, after setting all properties.
function TriggerPort_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TriggerPort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
