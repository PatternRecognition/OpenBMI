function varargout = p300_paradigm(varargin)
% P300_PARADIGM MATLAB code for p300_paradigm.fig
%      P300_PARADIGM, by itself, creates a new P300_PARADIGM or raises the existing
%      singleton*.
%
%      H = P300_PARADIGM returns the handle to a new P300_PARADIGM or the handle to
%      the existing singleton*.
%
%      P300_PARADIGM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in P300_PARADIGM.M with the given input arguments.
%
%      P300_PARADIGM('Property','Value',...) creates a new P300_PARADIGM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before p300_paradigm_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to p300_paradigm_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% % See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help p300_paradigm

% Last Modified by GUIDE v2.5 08-Aug-2017 13:59:53

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @p300_paradigm_OpeningFcn, ...
                   'gui_OutputFcn',  @p300_paradigm_OutputFcn, ...
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if you want to edit, edit from below this line 
% Oyeon Kwon, 2017.08
% oy_kwon@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes just before p300_paradigm is made visible.
function p300_paradigm_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to p300_paradigm (see VARARGIN)
set(gcf,'units','points','position',[500 150 495 460])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
set(handles.spellerText,'String','KOREA_UNIVERSITY');
set(handles.NumberOfSeq  ,'String','10');
set(handles.screenNumber ,'String','2');
set(handles.stimulusTime ,'String','0.135');
set(handles.IntervalTime ,'String','0.05');
set(handles.portNumber ,'String','D010');
set(handles.tcpiptxt ,'String','12300');
% set(handles.testText,'text40','off');
% set(handles.testText,'visible','off');
% set(handles.testText,'String','OPENBMI_SPELLER');
set(handles.notiontxt,'String','You have to start with selecting the p300 spller type');
scrPanelOn(handles,'off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose default command line output for p300_paradigm
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% UIWAIT makes p300_paradigm wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function initialize_gui(fig_handle, handles, isreset)
% If the metricdata field is present and the reset flag is false, it means
% we are we are just re-initializing a GUI by calling it from the cmd line
% while it is up. So, bail out as we dont want to reset the data.
if isfield(handles, 'metricdata') && ~isreset
    return;
end
% RESET(handles, true);
% Update handles structure
guidata(handles.figure1, handles);


% --- Outputs from this function are returned to the command line.
function varargout = p300_paradigm_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in makeP300paradigm.
function makeP300paradigm_Callback(hObject, eventdata, handles)
% hObject    handle to makeP300paradigm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.notiontxt,'String','Executing paradigm...'); drawnow;
% get offline or online
if get(handles.offbtn,'Value')
    exp_type = -1;
elseif get(handles.onbtn,'Value')
    exp_type = 0;
end

% get screensize info
ScreenSize = cellstr(get(handles.screenSize,'String'));
selectScreenSize = ScreenSize{get(handles.screenSize,'Value')};
if strcmp(cellstr(selectScreenSize),'Full screen')
    selectScreenSizeFirst = 'Full screen';   selectScreenSizeEnd=[];
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

% get parameters info
spellerText = get(handles.spellerText,'String');
NumberOfSeq = str2double(get(handles.NumberOfSeq,'String'));
screenNumber = str2double(get(handles.screenNumber,'String'));
stimulusTime = str2double(get(handles.stimulusTime,'String'));
IntervalTime = str2double(get(handles.IntervalTime,'String'));
portNumber = get(handles.portNumber,'String');

% warning box
if isempty(portNumber), error('No input port information'), end

% play paradigm
if ischar(selectScreenSizeFirst) && strcmp(selectScreenSizeFirst,'Full screen')
    if get(handles.random_sp,'Value')
                 gg =   random_speller({'exp_type',exp_type; 'port',portNumber;...
    'text',spellerText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
    'sti_Interval',IntervalTime});
    elseif get(handles.face_sp,'Value')
                     gg = face_speller({'exp_type',exp_type; 'port',portNumber;...
    'text',spellerText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
    'sti_Interval',IntervalTime});
    elseif get(handles.RC_sp,'Value')
                    gg = rc_speller({'exp_type',exp_type; 'port',portNumber;...
    'text',spellerText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
    'sti_Interval',IntervalTime});
    end
else
    if get(handles.random_sp,'Value')
                 gg =   random_speller({'exp_type',exp_type;  'port',portNumber;...
    'text',spellerText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
    'sti_Interval',IntervalTime;'screenSize',[selectScreenSizeFirst,selectScreenSizeEnd]});
    elseif get(handles.face_sp,'Value')
              gg = face_speller({'exp_type',exp_type; 'port',portNumber;...
    'text',spellerText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
    'sti_Interval',IntervalTime;'screenSize',[selectScreenSizeFirst,selectScreenSizeEnd]});
    elseif get(handles.RC_sp,'Value')
                   gg = rc_speller({'exp_type',exp_type; 'port',portNumber;...
    'text',spellerText;'nSequence',NumberOfSeq;'screenNum',screenNumber;'sti_Times',stimulusTime;...
    'sti_Interval',IntervalTime;'screenSize',[selectScreenSizeFirst,selectScreenSizeEnd]});
    end
end

set(handles.notiontxt,'String',gg);

initialize_gui(hObject, handles, false);

% --- Executes on button press in RC_sp.
function RC_sp_Callback(hObject, eventdata, handles)
% hObject    handle to RC_sp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.RC_sp,'Value'))
    set(handles.random_sp,'Value',0)
    set(handles.face_sp,'Value',0)
end
set(handles.notiontxt,'String','Now you should make your p300 paradigm then start');
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
set(handles.notiontxt,'String','Now you should make your p300 paradigm then start');
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
set(handles.notiontxt,'String','Now you should make your p300 paradigm then start');
% Hint: get(hObject,'Value') returns toggle state of face_sp


function scrPanelOn(handles, onoff)
set(handles.writeScreenSizeFirst,'visible',onoff);
set(handles.text29,'visible',onoff);
set(handles.writeScreenSizeEnd,'visible',onoff);
set(handles.screenSizeComplete,'visible',onoff);

% --- Executes on selection change in screenSize.
function screenSize_Callback(hObject, eventdata, handles)
% hObject    handle to screenSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
switch get(handles.screenSize,'value')
    case 1
        scrPanelOn(handles, 'off');
    case 2
        scrPanelOn(handles, 'off');
    case 3
        scrPanelOn(handles, 'off');
    case 4
        scrPanelOn(handles, 'off');
    case 5
       scrPanelOn(handles, 'off');
    case 6
        scrPanelOn(handles, 'on');    
end


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



function writeScreenSizeEnd_Callback(hObject, eventdata, handles)
% hObject    handle to writeScreenSizeEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of writeScreenSizeEnd as text
%        str2double(get(hObject,'String')) returns contents of writeScreenSizeEnd as a double
input = get(handles.writeScreenSizeEnd,'string');
if regexp(input, '[^0-9]')
    set(handles.notiontxt,'String','Screen size input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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



function writeScreenSizeFirst_Callback(hObject, eventdata, handles)
% hObject    handle to writeScreenSizeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of writeScreenSizeFirst as text
%        str2double(get(hObject,'String')) returns contents of writeScreenSizeFirst as a double
input = get(handles.writeScreenSizeFirst,'string');
if regexp(input, '[^0-9]')
    set(handles.notiontxt,'String','Screen size input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end



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



function spellerText_Callback(hObject, eventdata, handles)
% hObject    handle to spellerText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of spellerText as text
%        str2double(get(hObject,'String')) returns contents of spellerText as a double
input = get(handles.spellerText,'string');
if regexp(input, '[^A-Z.,0-9._ ;]')
    set(handles.notiontxt,'String','Text input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

    
% --- Executes during object creation, after setting all properties.
function spellerText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to spellerText (see GCBO)
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
input = get(handles.NumberOfSeq,'string');
if regexp(input, '[^0-9]')
    set(handles.notiontxt,'String','Sequence input format should be checked');
else
        set(handles.notiontxt,'String','Set');
%         set(handles.NumberOfSeq  ,'String','10');
end


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
input = get(handles.screenNumber,'string');
if regexp(input, '[^0-9]')
    set(handles.notiontxt,'String','Screen number input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.stimulusTime,'string');
if regexp(input, '[^0-9.,;]')
    set(handles.notiontxt,'String','Stimulus time input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.IntervalTime,'string');
if regexp(input, '[^0-9.,;]')
    set(handles.notiontxt,'String','Interval time input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
% input = get(handles.IntervalTime,'string');
% if regexp(input, '[!@#$%^&*()-=~+]')
%     set(handles.notiontxt,'String','Interval time input format should be checked');
% else
%         set(handles.notiontxt,'String','Set');
% end

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



function edit29_Callback(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit29 as text
%        str2double(get(hObject,'String')) returns contents of edit29 as a double


% --- Executes during object creation, after setting all properties.
function edit29_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit30_Callback(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit30 as text
%        str2double(get(hObject,'String')) returns contents of edit30 as a double


% --- Executes during object creation, after setting all properties.
function edit30_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in offbtn.
function offbtn_Callback(hObject, eventdata, handles)
% hObject    handle to offbtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.spellerText,'String','KOREA_UNIVERSITY');

if(get(handles.offbtn,'Value'))
    set(handles.onbtn,'Value',0)
end
set(handles.text41,'visible','off');
set(handles.tcpiptxt,'visible','off');
set(handles.servercheck,'visible','off');
set(handles.pushbutton12,'visible','on');
set(handles.notiontxt,'String','Set up for offline paradigm');
% Hint: get(hObject,'Value') returns toggle state of offbtn


% --- Executes on button press in onbtn.
function onbtn_Callback(hObject, eventdata, handles)
% hObject    handle to onbtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.onbtn,'Value'))
    set(handles.offbtn,'Value',0)
end
set(handles.spellerText,'String','OPENBMI_SPELLER');
% Hint: get(hObject,'Value') returns toggle state of onbtn
set(handles.text41,'visible','on');
set(handles.tcpiptxt,'visible','on');
set(handles.servercheck,'visible','on');
set(handles.pushbutton12,'visible','on');
set(handles.notiontxt,'String','Set up for online paradigm');

function testText_Callback(hObject, eventdata, handles)
% hObject    handle to testText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of testText as text
%        str2double(get(hObject,'String')) returns contents of testText as a double


% --- Executes during object creation, after setting all properties.
function testText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to testText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in servercheck.
function servercheck_Callback(hObject, eventdata, handles)
% hObject    handle to servercheck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock IO_LIB IO_ADD

% if isequal(get(handles.pushbutton12, 'string'), 'Brainvision check')
%     set(handles.notiontxt,'String','have to set up brainvision port first...'); drawnow;
%     return;
% end
% set(handles.notiontxt,'String','waiting check connection of TCPIP');
% bbci_acquire_bv('close');
set(handles.notiontxt,'String','Waiting check TCPIP connection...'); drawnow;
port = get(handles.portNumber ,'String');
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec(port);
tcpipp = str2double(get(handles.tcpiptxt,'String'));
% param = struct;
% state = bbci_acquire_bv('Init',param);
sock = tcpip('0.0.0.0',tcpipp ,'NetworkRole','server','timeout',3);  %
fopen(sock);
disp(1);
flushinput(sock);
while(true)
    a = fread(sock,19);    
    if ~isempty(a) 
        break;
    end
end

ppWrite(IO_ADD,a);  % 19 check
set(handles.notiontxt,'String','Check connection and start paradigm');

function tcpiptxt_Callback(hObject, eventdata, handles)
% hObject    handle to tcpiptxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tcpiptxt as text
%        str2double(get(hObject,'String')) returns contents of tcpiptxt as a double


% --- Executes during object creation, after setting all properties.
function tcpiptxt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tcpiptxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu4.
function popupmenu4_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu4


% --- Executes during object creation, after setting all properties.
function popupmenu4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit33_Callback(hObject, eventdata, handles)
% hObject    handle to edit33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit33 as text
%        str2double(get(hObject,'String')) returns contents of edit33 as a double


% --- Executes during object creation, after setting all properties.
function edit33_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit34_Callback(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit34 as text
%        str2double(get(hObject,'String')) returns contents of edit34 as a double


% --- Executes during object creation, after setting all properties.
function edit34_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit35_Callback(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit35 as text
%        str2double(get(hObject,'String')) returns contents of edit35 as a double


% --- Executes during object creation, after setting all properties.
function edit35_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit36_Callback(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit36 as text
%        str2double(get(hObject,'String')) returns contents of edit36 as a double


% --- Executes during object creation, after setting all properties.
function edit36_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit37_Callback(hObject, eventdata, handles)
% hObject    handle to edit37 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit37 as text
%        str2double(get(hObject,'String')) returns contents of edit37 as a double


% --- Executes during object creation, after setting all properties.
function edit37_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit37 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit38_Callback(hObject, eventdata, handles)
% hObject    handle to edit38 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit38 as text
%        str2double(get(hObject,'String')) returns contents of edit38 as a double


% --- Executes during object creation, after setting all properties.
function edit38_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit38 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit39_Callback(hObject, eventdata, handles)
% hObject    handle to edit39 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit39 as text
%        str2double(get(hObject,'String')) returns contents of edit39 as a double


% --- Executes during object creation, after setting all properties.
function edit39_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit39 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% % --- Executes on button press in offbtn.
% function offlinebtn_Callback(hObject, eventdata, handles)
% % hObject    handle to offbtn (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hint: get(hObject,'Value') returns toggle state of offbtn
% set(handles.text41,'visible','off');
% set(handles.tcpiptxt,'visible','off');
% set(handles.servercheck,'visible','off');
% 
% 
% % --- Executes on button press in onbtn.
% function onlinebtn_Callback(hObject, eventdata, handles)
% % hObject    handle to onbtn (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hint: get(hObject,'Value') returns toggle state of onbtn
% set(handles.text41,'visible','on');
% set(handles.tcpiptxt,'visible','on');
% set(handles.servercheck,'visible','on');
% 

% --- Executes on button press in finishbtn.
function finishbtn_Callback(hObject, eventdata, handles)
% hObject    handle to finishbtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.notiontxt,'String','Check brainvison port (or TCPIP for online access)');


% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global IO_LIB IO_ADD
% bbci_acquire_bv('close');
port = get(handles.portNumber ,'String');
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec(port);
ppWrite(IO_ADD,19);  % 19 check
set(handles.notiontxt,'String','Check trigger... "19" in your brainvision');



% --- Executes on button press in Reset.
function Reset_Callback(hObject, eventdata, handles)
% hObject    handle to Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
RESET(handles, false);

function RESET(handles, init)
if ~init
    a = [500 150 495 460];
else
    a = get(handles.figure1,'Position');
end
% a(4) = 450;
set(handles.notiontxt,'String','Reset all parameters...'); drawnow;
pause(2);
set(handles.figure1,'Position',a);
set(handles.spellerText,'String','KOREA_UNIVERSITY');
set(handles.NumberOfSeq  ,'String','10');
set(handles.screenNumber ,'String','2');
set(handles.stimulusTime ,'String','0.135');
set(handles.IntervalTime ,'String','0.05');
set(handles.portNumber ,'String','D010');
set(handles.tcpiptxt ,'String','12300');
% set(handles.testText,'text40','off');
% set(handles.testText,'visible','off');
% set(handles.testText,'String','OPENBMI_SPELLER');
scrPanelOn(handles,'off');
set(handles.RC_sp,'Value',0);
set(handles.random_sp,'Value',0);
set(handles.face_sp,'Value',0);
set(handles.offbtn,'Value',0);
set(handles.onbtn,'Value',0);
set(handles.text41,'visible','on');
set(handles.tcpiptxt,'visible','on');
set(handles.servercheck,'visible','on');
set(handles.pushbutton12,'visible','on');
set(handles.notiontxt,'String','You have to start with selecting the p300 spller type');
set(handles.screenSize,'value',1);

global state sock
bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
disp('will close to connection');
try
    fclose(sock);
catch
end
