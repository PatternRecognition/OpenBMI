function varargout = paradigm_p300(varargin)
% PARADIGM_P300 MATLAB code for paradigm_p300.fig
%      PARADIGM_P300, by itself, creates a new PARADIGM_P300 or raises the existing
%      singleton*.
%
%      H = PARADIGM_P300 returns the handle to a new PARADIGM_P300 or the handle to
%      the existing singleton*.
%
%      PARADIGM_P300('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PARADIGM_P300.M with the given input arguments.
%
%      PARADIGM_P300('Property','Value',...) creates a new PARADIGM_P300 or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before paradigm_p300_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to paradigm_p300_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help paradigm_p300

% Last Modified by GUIDE v2.5 08-Nov-2016 18:53:29

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @paradigm_p300_OpeningFcn, ...
                   'gui_OutputFcn',  @paradigm_p300_OutputFcn, ...
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

% --- Executes just before paradigm_p300 is made visible.
function paradigm_p300_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to paradigm_p300 (see VARARGIN)
set(gcf,'units','points','position',[350 200 500 500])
% Choose default command line output for paradigm_p300
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

initialize_gui(hObject, handles, false);

% UIWAIT makes paradigm_p300 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = paradigm_p300_OutputFcn(hObject, eventdata, handles)
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


% --- Executes during object creation, after setting all properties.
function calX_CreateFcn(hObject, eventdata, handles)
% hObject    handle to calX (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function trainingText_Callback(hObject, eventdata, handles)
% hObject    handle to trainingText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of trainingText as text
%        str2double(get(hObject,'String')) returns contents of trainingText as a double


% --- Executes during object creation, after setting all properties.
function trainingText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to trainingText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function NumberOfSeq_Callback(hObject, eventdata, handles)
% hObject    handle to NumberOfSeq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NumberOfSeq as text
%        str2double(get(hObject,'String')) returns contents of NumberOfSeq as a double


% --- Executes during object creation, after setting all properties.
function NumberOfSeq_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NumberOfSeq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function screenNumber_Callback(hObject, eventdata, handles)
% hObject    handle to screenNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of screenNumber as text
%        str2double(get(hObject,'String')) returns contents of screenNumber as a double


% --- Executes during object creation, after setting all properties.
function screenNumber_CreateFcn(hObject, eventdata, handles)
% hObject    handle to screenNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function stimulusTime_Callback(hObject, eventdata, handles)
% hObject    handle to stimulusTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stimulusTime as text
%        str2double(get(hObject,'String')) returns contents of stimulusTime as a double


% --- Executes during object creation, after setting all properties.
function stimulusTime_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stimulusTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function IntervalTime_Callback(hObject, eventdata, handles)
% hObject    handle to IntervalTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of IntervalTime as text
%        str2double(get(hObject,'String')) returns contents of IntervalTime as a double


% --- Executes during object creation, after setting all properties.
function IntervalTime_CreateFcn(hObject, eventdata, handles)
% hObject    handle to IntervalTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function portNumber_Callback(hObject, eventdata, handles)
% hObject    handle to portNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of portNumber as text
%        str2double(get(hObject,'String')) returns contents of portNumber as a double


% --- Executes during object creation, after setting all properties.
function portNumber_CreateFcn(hObject, eventdata, handles)
% hObject    handle to portNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function targetTrigger_Callback(hObject, eventdata, handles)
% hObject    handle to targetTrigger (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of targetTrigger as text
%        str2double(get(hObject,'String')) returns contents of targetTrigger as a double

% --- Executes during object creation, after setting all properties.
function targetTrigger_CreateFcn(hObject, eventdata, handles)
% hObject    handle to targetTrigger (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function nontargetTrigger_Callback(hObject, eventdata, handles)
% hObject    handle to nontargetTrigger (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nontargetTrigger as text
%        str2double(get(hObject,'String')) returns contents of nontargetTrigger as a double

% --- Executes during object creation, after setting all properties.
function nontargetTrigger_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nontargetTrigger (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%%
% --- Executes on button press in makeP300paradigm.
function makeP300paradigm_Callback(hObject, eventdata, handles)
% hObject    handle to makeP300paradigm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

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

trainingText = get(handles.trainingText,'String');
NumberOfSeq = str2double(get(handles.NumberOfSeq,'String'));
screenNumber = str2double(get(handles.screenNumber,'String'));
stimulusTime = str2double(get(handles.stimulusTime,'String'));
IntervalTime = str2double(get(handles.IntervalTime,'String'));
portNumber = get(handles.portNumber,'String');
targetTrigger = str2double(get(handles.targetTrigger,'String'));
nontargetTrigger = str2double(get(handles.nontargetTrigger,'String'));



% warning box
if isempty(portNumber), error('No input port information'), end
% set defaults
if isempty(trainingText)||sum(isnan(trainingText))
    trainingText='DEFAULT_TEXT';
end
if isempty(NumberOfSeq)||isnan(NumberOfSeq),
    NumberOfSeq=10;
end
if isempty(screenNumber)||isnan(screenNumber)
    screenNumber=2;
end
if isempty(stimulusTime)||isnan(stimulusTime)
    stimulusTime=0.135;
end
if isempty(IntervalTime)||isnan(IntervalTime)
    IntervalTime=0.05;
end
if isempty(targetTrigger)||isnan(targetTrigger)
    targetTrigger=1;
end
if isempty(nontargetTrigger)||isnan(nontargetTrigger)
    nontargetTrigger=2;
end


if ischar(selectScreenSizeFirst) && strcmp(selectScreenSizeFirst,'Full screen')
    speller_offline(exp_type,{'port',portNumber;...
    'text',trainingText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
    'sti_Interval',IntervalTime;'trigger',[targetTrigger,nontargetTrigger]})
else
    speller_offline(exp_type,{'port',portNumber;'screenSize',[selectScreenSizeFirst,selectScreenSizeEnd];...
        'text',trainingText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
        'sti_Interval',IntervalTime;'trigger',[targetTrigger,nontargetTrigger]})
end

handles.exp_type = exp_type;
handles.selectScreenSizeFirst = selectScreenSizeFirst;
handles.selectScreenSizeEnd = selectScreenSizeEnd;
handles.trainingText = trainingText;
handles.NumberOfSeq = NumberOfSeq;
handles.screenNumber = screenNumber;
handles.stimulusTime = stimulusTime;
handles.IntervalTime = IntervalTime;
handles.portNumber= portNumber;
handles.targetTrigger=targetTrigger;
handles.nontargetTrigger=nontargetTrigger;
guidata(hObject, handles)

% --- Executes on button press in saveParaPmter.
function saveParaPmter_Callback(hObject, eventdata, handles)
% hObject    handle to saveParaPmter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% global exp_type selectScreenSizeFirst selectScreenSizeEnd trainingText NumberOfSeq...
%     screenNumber stimulusTime IntervalTime portNumber targetTrigger nontargetTrigger
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

trainingText = get(handles.trainingText,'String');
NumberOfSeq = str2double(get(handles.NumberOfSeq,'String'));
screenNumber = str2double(get(handles.screenNumber,'String'));
stimulusTime = str2double(get(handles.stimulusTime,'String'));
IntervalTime = str2double(get(handles.IntervalTime,'String'));
portNumber = get(handles.portNumber,'String');
targetTrigger = str2double(get(handles.targetTrigger,'String'));
nontargetTrigger = str2double(get(handles.nontargetTrigger,'String'));



% warning box
if isempty(portNumber), error('No input port information'), end
% set defaults
if isempty(trainingText)||sum(isnan(trainingText))
    trainingText='DEFAULT_TEXT';
end
if isempty(NumberOfSeq)||isnan(NumberOfSeq),
    NumberOfSeq=10;
end
if isempty(screenNumber)||isnan(screenNumber)
    screenNumber=2;
end
if isempty(stimulusTime)||isnan(stimulusTime)
    stimulusTime=0.135;
end
if isempty(IntervalTime)||isnan(IntervalTime)
    IntervalTime=0.05;
end
if isempty(targetTrigger)||isnan(targetTrigger)
    targetTrigger=1;
end
if isempty(nontargetTrigger)||isnan(nontargetTrigger)
    nontargetTrigger=2;
end

valParPmter.exp_type = exp_type;
valParPmter.selectScreenSizeFirst = selectScreenSizeFirst;
if ~strcmp(selectScreenSizeFirst,'Full screen')
valParPmter.selectScreenSizeEnd = selectScreenSizeEnd;
end
valParPmter.trainingText = trainingText;
valParPmter.NumberOfSeq = NumberOfSeq;
valParPmter.screenNumber = screenNumber;
valParPmter.stimulusTime = stimulusTime;
valParPmter.IntervalTime = IntervalTime;
valParPmter.portNumber = portNumber;
valParPmter.targetTrigger = targetTrigger;
valParPmter.nontargetTrigger = nontargetTrigger;


uisave('valParPmter','parameter')
% save('D:\GUI\paradigm_parameter', 'valParPmter');


% --- Executes on button press in ADD.
function ADD_Callback(hObject, eventdata, handles)
% hObject    handle to ADD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global EEG
[file,path]=uigetfile('*.eeg','Load EEG file (.eeg)');
[file]=strsplit(file,'.eeg');

prompt={'Enter trigger number and corresponding class name:','Enter sampling frequency:'};
defaultans = {'{''1'',''Target'';''2'',''Non-target''}','100'};
c=inputdlg(prompt,'Marker',[1 70],defaultans);
cls=c{1};
fs=str2double(c{2});
[EEG.data, EEG.marker, EEG.info]=Load_EEG([path,file{1}],{'device','brainVision';'marker',eval(cls);'fs',fs});


handles.EEG = EEG;
guidata(hObject, handles)

% --- Executes on button press in REMOVE.
function REMOVE_Callback(hObject, eventdata, handles)
% hObject    handle to REMOVE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str=sprintf('File name');
str2=sprintf(' ');
set(handles.DataName, 'String', str);
set(handles.DATALIST, 'String', str2);

function TimeIvalFirst_Callback(hObject, eventdata, handles)
% hObject    handle to TimeIvalFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeIvalFirst as text
%        str2double(get(hObject,'String')) returns contents of TimeIvalFirst as a double

% --- Executes during object creation, after setting all properties.
function TimeIvalFirst_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TimeIvalFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function TimeIvalEnd_Callback(hObject, eventdata, handles)
% hObject    handle to TimeIvalEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeIvalEnd as text
%        str2double(get(hObject,'String')) returns contents of TimeIvalEnd as a double

% --- Executes during object creation, after setting all properties.
function TimeIvalEnd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TimeIvalEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function chanSel_Callback(hObject, eventdata, handles)
% hObject    handle to chanSel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of chanSel as text
%        str2double(get(hObject,'String')) returns contents of chanSel as a double

% --- Executes during object creation, after setting all properties.
function chanSel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to chanSel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function baselineTimeFirst_Callback(hObject, eventdata, handles)
% hObject    handle to baselineTimeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baselineTimeFirst as text
%        str2double(get(hObject,'String')) returns contents of baselineTimeFirst as a double


% --- Executes during object creation, after setting all properties.
function baselineTimeFirst_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baselineTimeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function baselineTimeEnd_Callback(hObject, eventdata, handles)
% hObject    handle to baselineTimeEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baselineTimeEnd as text
%        str2double(get(hObject,'String')) returns contents of baselineTimeEnd as a double


% --- Executes during object creation, after setting all properties.
function baselineTimeEnd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baselineTimeEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function selectedTimeFirst_Callback(hObject, eventdata, handles)
% hObject    handle to selectedTimeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of selectedTimeFirst as text
%        str2double(get(hObject,'String')) returns contents of selectedTimeFirst as a double


% --- Executes during object creation, after setting all properties.
function selectedTimeFirst_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectedTimeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function selectedTimeEnd_Callback(hObject, eventdata, handles)
% hObject    handle to selectedTimeEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of selectedTimeEnd as text
%        str2double(get(hObject,'String')) returns contents of selectedTimeEnd as a double


% --- Executes during object creation, after setting all properties.
function selectedTimeEnd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectedTimeEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function numFeature_Callback(hObject, eventdata, handles)
% hObject    handle to numFeature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of numFeature as text
%        str2double(get(hObject,'String')) returns contents of numFeature as a double


% --- Executes during object creation, after setting all properties.
function numFeature_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numFeature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in makeClassifier.
function makeClassifier_Callback(hObject, eventdata, handles)
% hObject    handle to makeClassifier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global EEG clf_param ch_idx 
EEG = handles.EEG;

chanSel = get(handles.chanSel,'String');
TimeIvalFirst = str2double(get(handles.TimeIvalFirst,'String'));
TimeIvalEnd = str2double(get(handles.TimeIvalEnd,'String'));
baselineTimeFirst = str2double(get(handles.baselineTimeFirst,'String'));
baselineTimeEnd = str2double(get(handles.baselineTimeEnd,'String'));
selectedTimeFirst = str2double(get(handles.selectedTimeFirst,'String'));
selectedTimeEnd = str2double(get(handles.selectedTimeEnd,'String'));
numFeature = str2double(get(handles.numFeature,'String'));
% numFeature = double(handles.numFeature);

TimeIval = [TimeIvalFirst TimeIvalEnd];
baselineTime = [baselineTimeFirst baselineTimeEnd];
selectedTime = [selectedTimeFirst selectedTimeEnd];

[clf_param,ch_idx] = p300_classifier(EEG,{'segTime',TimeIval;'baseTime',baselineTime;'selTime',selectedTime;'nFeature',numFeature;'channel',chanSel})


handles.TimeIval = TimeIval;
handles.selectedTime = selectedTime;
handles.numFeature= numFeature;
handles.baselineTime = baselineTime;
handles.ch_idx = ch_idx;
guidata(hObject, handles)

% --- Executes on button press in Connect.
function Connect_Callback(hObject, eventdata, handles)
% hObject    handle to Connect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global clf_param ch_idx TimeIval selectedTime numFeature baselineTime

NumberOfSeq=str2double(get(handles.NumberOfSeq,'String'));

TimeIval = handles.TimeIval;
selectedTime = handles.selectedTime;
numFeature = handles.numFeature;
baselineTime = handles.baselineTime;
ch_idx = handles.ch_idx;

p300_client( clf_param, {'segTime',TimeIval;'baseTime',baselineTime;'selTime',selectedTime;'nFeature',numFeature;'channel',ch_idx;'nSequence',NumberOfSeq} )
