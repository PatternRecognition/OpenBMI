function varargout = Online_p300(varargin)
% ONLINE_P300 MATLAB code for Online_p300.fig
%      ONLINE_P300, by itself, creates a new ONLINE_P300 or raises the existing
%      singleton*.
%
%      H = ONLINE_P300 returns the handle to a new ONLINE_P300 or the handle to
%      the existing singleton*.
%
%      ONLINE_P300('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ONLINE_P300.M with the given input arguments.
%
%      ONLINE_P300('Property','Value',...) creates a new ONLINE_P300 or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Online_p300_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Online_p300_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Online_p300

% Last Modified by GUIDE v2.5 09-Nov-2016 14:28:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Online_p300_OpeningFcn, ...
                   'gui_OutputFcn',  @Online_p300_OutputFcn, ...
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

% --- Executes just before Online_p300 is made visible.
function Online_p300_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Online_p300 (see VARARGIN)
set(gcf,'units','points','position',[350 200 470 350])
% Choose default command line output for Online_p300
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

initialize_gui(hObject, handles, false);

% UIWAIT makes Online_p300 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Online_p300_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function initialize_gui(fig_handle, handles, isreset)
% If the metricdata field is present and the reset flag is false, it means
% we are we are just re-initializing a GUI by calling it from the cmd line
% while it is up. So, bail out as we dont want to reset the data.
if isfield(handles, 'metricdata') && ~isreset
    return;
end
set(handles.writeScreenSizeFirst,'visible','off');
set(handles.writeScreenSizeEnd,'visible','off');
set(handles.calX,'visible','off');

% Update handles structure
guidata(handles.figure1, handles);


% --- Executes on button press in loadtrainpamter.
function loadtrainpamter_Callback(hObject, eventdata, handles)
% hObject    handle to loadtrainpamter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global valParPmter
[file,path]=uigetfile('*.mat','Load parameters');
load(sprintf('%s%s',path,file))
if(get(handles.loadtrainpamter,'Value'))
    set(handles.newParameter,'Value',0)
    set(handles.numofSequence,'String', num2str(valParPmter.NumberOfSeq))
    set(handles.screenNum,'String', num2str(valParPmter.screenNumber))
    set(handles.stimtime,'String', num2str(valParPmter.stimulusTime))
    set(handles.ivalTime,'String', num2str(valParPmter.IntervalTime))
    set(handles.portNum,'String', valParPmter.portNumber)
    set(handles.ivalTime,'String', num2str(valParPmter.IntervalTime))
%     set(handles.targetNum,'String', num2str(valParPmter.targetTrigger))
%     set(handles.targetNum,'String', num2str(valParPmter.targetTrigger))
%     set(handles.nonTargetnum,'String', num2str(valParPmter.nontargetTrigger))

    set(handles.screenSize,'visible', 'off')
    set(handles.writeScreenSizeFirst,'visible', 'on');
    set(handles.calX,'visible', 'on');
    set(handles.writeScreenSizeFirst,'String', num2str(valParPmter.selectScreenSizeFirst));
    if ~strcmp(valParPmter.selectScreenSizeFirst,'Full screen')
        set(handles.writeScreenSizeEnd,'visible', 'on');
        set(handles.writeScreenSizeEnd,'String', num2str(valParPmter.selectScreenSizeEnd));
    end
    
    switch valParPmter.exp_type
        case 'random'
            set(handles.random_sp,'Value',1)
        case 'RC'
            set(handles.RC_sp,'Value',1)
        case 'face'
            set(handles.face_sp,'Value',1)
    end
    
end

handles.valParPmter = valParPmter;
guidata(hObject, handles)


% Hint: get(hObject,'Value') returns toggle state of loadtrainpamter


% --- Executes on button press in newParameter.
function newParameter_Callback(hObject, eventdata, handles)
% hObject    handle to newParameter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.newParameter,'Value'))
    set(handles.loadtrainpamter,'Value',0)
    set(handles.numofSequence,'String', '5')
    set(handles.screenNum,'String', '2')
    set(handles.stimtime,'String', '0.135')
    set(handles.ivalTime,'String', '0.05')
    set(handles.portNum,'String', '5FF8')
    set(handles.hostnum,'String', 'localhost');
    
    set(handles.screenSize,'visible', 'on')
    set(handles.writeScreenSizeFirst,'visible', 'off');
    set(handles.calX,'visible', 'off');
    set(handles.writeScreenSizeEnd,'visible', 'off');
    set(handles.writeScreenSizeFirst,'String', ' ');
    set(handles.writeScreenSizeEnd,'String',  ' ');
end
% Hint: get(hObject,'Value') returns toggle state of newParameter


% --- Executes on selection change in screenSize.
function screenSize_Callback(hObject, eventdata, handles)
% hObject    handle to screenSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns screenSize contents as cell array
%        contents{get(hObject,'Value')} returns selected item from screenSize


% --- Executes during object creation, after setting all properties.
function screenSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to screenSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function writeScreenSizeFirst_Callback(hObject, eventdata, handles)
% hObject    handle to writeScreenSizeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of writeScreenSizeFirst as text
%        str2double(get(hObject,'String')) returns contents of writeScreenSizeFirst as a double


% --- Executes during object creation, after setting all properties.
function writeScreenSizeFirst_CreateFcn(hObject, eventdata, handles)
% hObject    handle to writeScreenSizeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function writeScreenSizeEnd_Callback(hObject, eventdata, handles)
% hObject    handle to writeScreenSizeEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of writeScreenSizeEnd as text
%        str2double(get(hObject,'String')) returns contents of writeScreenSizeEnd as a double


% --- Executes during object creation, after setting all properties.
function writeScreenSizeEnd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to writeScreenSizeEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in screenSizeComplete.
function screenSizeComplete_Callback(hObject, eventdata, handles)
% hObject    handle to screenSizeComplete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
screenSize = cellstr(get(handles.screenSize,'String'));
if strcmp(cellstr(screenSize{6}),'Write direct')
    set(handles.writeScreenSizeFirst,'visible','on');
    set(handles.writeScreenSizeEnd,'visible','on');
    set(handles.calX,'visible','on');
end


function testingText_Callback(hObject, eventdata, handles)
% hObject    handle to testingText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of testingText as text
%        str2double(get(hObject,'String')) returns contents of testingText as a double


% --- Executes during object creation, after setting all properties.
function testingText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to testingText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function numofSequence_Callback(hObject, eventdata, handles)
% hObject    handle to numofSequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of numofSequence as text
%        str2double(get(hObject,'String')) returns contents of numofSequence as a double


% --- Executes during object creation, after setting all properties.
function numofSequence_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numofSequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function screenNum_Callback(hObject, eventdata, handles)
% hObject    handle to screenNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of screenNum as text
%        str2double(get(hObject,'String')) returns contents of screenNum as a double


% --- Executes during object creation, after setting all properties.
function screenNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to screenNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function stimtime_Callback(hObject, eventdata, handles)
% hObject    handle to stimtime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stimtime as text
%        str2double(get(hObject,'String')) returns contents of stimtime as a double


% --- Executes during object creation, after setting all properties.
function stimtime_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stimtime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ivalTime_Callback(hObject, eventdata, handles)
% hObject    handle to ivalTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ivalTime as text
%        str2double(get(hObject,'String')) returns contents of ivalTime as a double


% --- Executes during object creation, after setting all properties.
function ivalTime_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ivalTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function portNum_Callback(hObject, eventdata, handles)
% hObject    handle to portNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of portNum as text
%        str2double(get(hObject,'String')) returns contents of portNum as a double


% --- Executes during object creation, after setting all properties.
function portNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to portNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function targetNum_Callback(hObject, eventdata, handles)
% hObject    handle to targetNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of targetNum as text
%        str2double(get(hObject,'String')) returns contents of targetNum as a double


% --- Executes during object creation, after setting all properties.
function targetNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to targetNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function nonTargetnum_Callback(hObject, eventdata, handles)
% hObject    handle to nonTargetnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nonTargetnum as text
%        str2double(get(hObject,'String')) returns contents of nonTargetnum as a double


% --- Executes during object creation, after setting all properties.
function nonTargetnum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nonTargetnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in RC_sp.
function RC_sp_Callback(hObject, eventdata, handles)
% hObject    handle to RC_sp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.RC_sp,'Value'))
    set(handles.random_sp,'Value',0)
    set(handles.face_sp,'Value',0)
end
% Hint: get(hObject,'Value') returns toggle state of RC_sp


% --- Executes on button press in random_sp.
function random_sp_Callback(hObject, eventdata, handles)
% hObject    handle to random_sp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.random_sp,'Value'))
    set(handles.RC_sp,'Value',0)
    set(handles.face_sp,'Value',0)
end
% Hint: get(hObject,'Value') returns toggle state of random_sp


% --- Executes on button press in face_sp.
function face_sp_Callback(hObject, eventdata, handles)
% hObject    handle to face_sp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.face_sp,'Value'))
    set(handles.random_sp,'Value',0)
    set(handles.RC_sp,'Value',0)
end
% Hint: get(hObject,'Value') returns toggle state of face_sp


% --- Executes on button press in Connect.
function Connect_Callback(hObject, eventdata, handles)
% hObject    handle to Connect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global valParPmter
if(get(handles.loadtrainpamter,'Value'))
%     load('D:\GUI\paradigm_parameter.mat');
    exp_type = valParPmter.exp_type;
    testingText = get(handles.testingText,'String');
    NumberOfSeq = valParPmter.NumberOfSeq;
    screenNumber = valParPmter.screenNumber;
    stimulusTime = valParPmter.stimulusTime;
    IntervalTime = valParPmter.IntervalTime;
    portNumber = valParPmter.portNumber;
    targetTrigger = valParPmter.targetTrigger;
    nontargetTrigger = valParPmter.nontargetTrigger;
    selectScreenSizeFirst = valParPmter.selectScreenSizeFirst;
%     selectScreenSizeEnd = valParPmter.selectScreenSizeEnd;
    if strcmp(valParPmter.selectScreenSizeFirst,'Full Screen')
        selectScreenSizeEnd = valParPmter.selectScreenSizeEnd;
    end
elseif(get(handles.newParameter,'Value'))
    
    if get(handles.random_sp,'Value')
        exp_type='random';
    elseif get(handles.face_sp,'Value')
        exp_type='face';
    elseif get(handles.RC_sp,'Value')
        exp_type='RC';
    end
    
    ScreenSize = cellstr(get(handles.screenSize,'String'));
    selectScreenSize = ScreenSize{get(handles.screenSize,'Value')};
    if strcmp(cellstr(selectScreenSize),'Full screen')
        selectScreenSizeFirst = 'Full screen';
    elseif strcmp(cellstr(selectScreenSize),'1920 x 1200')
        selectScreenSizeFirst = 1920; selectScreenSizeEnd = 1200;
    elseif strcmp(cellstr(selectScreenSize),'1920 x 1080')
        selectScreenSizeFirst = 1920; selectScreenSizeEnd = 1080;
    elseif strcmp(cellstr(selectScreenSize),'1280 x 1024')
        selectScreenSizeFirst = 1280; selectScreenSizeEnd = 1024;
    elseif strcmp(cellstr(selectScreenSize),' 800 x  600')
        selectScreenSizeFirst = 800; selectScreenSizeEnd = 600;
    elseif strcmp(cellstr(selectScreenSize),'Write direct')
        selectScreenSizeFirst = str2double(get(handles.writeScreenSizeFirst,'String'));
        selectScreenSizeEnd = str2double(get(handles.writeScreenSizeEnd,'String'));
    end
    
    testingText = get(handles.testingText,'String');
    NumberOfSeq = str2double(get(handles.numofSequence,'String'));
    screenNumber = str2double(get(handles.screenNum,'String'));
    stimulusTime = str2double(get(handles.stimtime,'String'));
    IntervalTime = str2double(get(handles.ivalTime,'String'));
    portNumber = get(handles.portNum,'String');
%     targetTrigger = str2double(get(handles.targetNum,'String'));
%     nontargetTrigger = str2double(get(handles.nonTargetnum,'String')); 
    hostnum = get(handles.hostnum,'String');
end

if ischar(selectScreenSizeFirst) && strcmp(selectScreenSizeFirst,'Full screen')
    p300_server(exp_type,{'port',portNumber;...
        'text',testingText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
        'sti_Interval',IntervalTime;'hostnum', hostnum});
else
    p300_server(exp_type,{'port',portNumber;'screenSize',[selectScreenSizeFirst,selectScreenSizeEnd];...
        'text',testingText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
        'sti_Interval',IntervalTime;'hostnum', hostnum});
end



function hostnum_Callback(hObject, eventdata, handles)
% hObject    handle to hostnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hostnum as text
%        str2double(get(hObject,'String')) returns contents of hostnum as a double


% --- Executes during object creation, after setting all properties.
function hostnum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hostnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
