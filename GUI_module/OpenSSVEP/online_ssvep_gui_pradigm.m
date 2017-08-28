function varargout = online_ssvep_gui_pradigm(varargin)
% ONLINE_SSVEP_GUI_PARADIGM MATLAB code for online_ssvep_gui_pradigm.fig
%      ONLINE_SSVEP_GUI_PARADIGM, by itself, creates a new ONLINE_SSVEP_GUI_PARADIGM or raises the existing
%      singleton*.
%
%      H = ONLINE_SSVEP_GUI_PARADIGM returns the handle to a new ONLINE_SSVEP_GUI_PARADIGM or the handle to
%      the existing singleton*.
%
%      ONLINE_SSVEP_GUI_PARADIGM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ONLINE_SSVEP_GUI_PARADIGM.M with the given input arguments.
%lc;
%      ONLINE_SSVEP_GUI_PARADIGM('Property','Value',...) creates a new ONLINE_SSVEP_GUI_PARADIGM or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before online_ssvep_gui_pradigm_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to online_ssvep_gui_pradigm_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help online_ssvep_gui_pradigm

% Last Modified by GUIDE v2.5 25-Aug-2017 15:38:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @online_ssvep_gui_pradigm_OpeningFcn, ...
    'gui_OutputFcn',  @online_ssvep_gui_pradigm_OutputFcn, ...
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

% --- Executes just before online_ssvep_gui_pradigm is made visible.
function online_ssvep_gui_pradigm_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to online_ssvep_gui_pradigm (see VARARGIN)

% Choose default command line output for online_ssvep_gui_pradigm
% save('InitData.mat','handles');
handles.output = hObject;
%
global IO_ADDR IO_LIB sock;
% Update handles structure
guidata(hObject, handles);
initialize_gui(hObject, handles, false);

% UIWAIT makes online_ssvep_gui_pradigm wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = online_ssvep_gui_pradigm_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

function initialize_gui(fig_handle, handles, isreset)
% If the metricdata field is present and the reset flag is false, it means
% we are we are just re-initializing a GUI by calling it from the cmd line
% while it is up. So, bail out as we dont want to reset the data.

RESET(handles);
% Update handles structure
guidata(handles.figure1, handles);


function scrRow_Callback(hObject, eventdata, handles)
% hObject    handle to scrRow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of scrRow as text
%        str2double(get(hObject,'String')) returns contents of scrRow as a double
input = get(handles.scrRow,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
    set(handles.scrRow,'String','1920');
end

% --- Executes during object creation, after setting all properties.
function scrRow_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scrRow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function scrCol_Callback(hObject, eventdata, handles)
% hObject    handle to scrCol (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of scrCol as text
%        str2double(get(hObject,'String')) returns contents of scrCol as a double
input = get(handles.scrCol,'String');
if regexp(input, '[^0-9]')
    nset(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
    set(handles.scrCol,'String','1080');
end

% --- Executes during object creation, after setting all properties.
function scrCol_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scrCol (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function scrNum_Callback(hObject, eventdata, handles)
% hObject    handle to scrNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of scrNum as text
%        str2double(get(hObject,'String')) returns contents of scrNum as a double
input = get(handles.scrNum,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
    set(handles.scrNum,'String','2');
end

% --- Executes during object creation, after setting all properties.
function scrNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scrNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function numTrials_Callback(hObject, eventdata, handles)
% hObject    handle to numTrials (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of numTrials as text
%        str2double(get(hObject,'String')) returns contents of numTrials as a double
input = get(handles.numTrials,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String',sprintf('[%s] is not acceptable', input));
    set(handles.numTrials,'String','10');
end

% --- Executes during object creation, after setting all properties.
function numTrials_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numTrials (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function freq_Callback(hObject, eventdata, handles)
% hObject    handle to freq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freq as text
%        str2double(get(hObject,'String')) returns contents of freq as a double
input = get(handles.freq,'String');
if regexp(input, '[^0-9., ;]')
    set(handles.noti_text, 'String',sprintf('[%s] is not acceptable', input));
    set(handles.freq,'String','6.67, 7.5, 8.57,10, 12');
end

% --- Executes during object creation, after setting all properties.
function freq_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function timeSti_Callback(hObject, eventdata, handles)
% hObject    handle to timeSti (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of timeSti as text
%        str2double(get(hObject,'String')) returns contents of timeSti as a double
input = get(handles.timeSti,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String',sprintf('[%s] is not acceptable', input));
    set(handles.timeSti,'String','5');
end

% --- Executes during object creation, after setting all properties.
function timeSti_CreateFcn(hObject, eventdata, handles)
% hObject    handle to timeSti (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function boxSize_Callback(hObject, eventdata, handles)
% hObject    handle to boxSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of boxSize as text
%        str2double(get(hObject,'String')) returns contents of boxSize as a double
input = get(handles.boxSize,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String',sprintf('[%s] is not acceptable', input));
    set(handles.boxSize,'String','150');
end

% --- Executes during object creation, after setting all properties.
function boxSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to boxSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function btwBox_Callback(hObject, eventdata, handles)
% hObject    handle to btwBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of btwBox as text
%        str2double(get(hObject,'String')) returns contents of btwBox as a double
input = get(handles.btwBox,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String',sprintf('[%s] is not acceptable', input));
    set(handles.btwBox,'String','200');
end

% --- Executes during object creation, after setting all properties.
function btwBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to btwBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function timeRest_Callback(hObject, eventdata, handles)
% hObject    handle to timeRest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of timeRest as text
%        str2double(get(hObject,'String')) returns contents of timeRest as a double
input = get(handles.timeRest,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String',sprintf('[%s] is not acceptable', input));
    set(handles.timeRest,'String','2');
end

% --- Executes during object creation, after setting all properties.
function timeRest_CreateFcn(hObject, eventdata, handles)
% hObject    handle to timeRest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in btnReset.
function btnReset_Callback(hObject, eventdata, handles)
% hObject    handle to btnReset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% disp(get(handles.btwBox,'String'));
% load('InitData.mat');
% handles.output = hObject;
% disp(get(handles.btwBox,'String'));
% guidata(handles.figure1, handles);
RESET(handles);

% --- Executes on button press in btnStart.
function btnStart_Callback(hObject, eventdata, handles)
% hObject    handle to btnStart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock
try
    switch get(handles.scrPop,'value')
        case 1
            scrSize = [0, 0, 0, 0];
        case 2
            scrSize = [0, 0, 1920, 1200];
        case 3
            scrSize = [0, 0, 1920, 1080];
        case 4
            scrSize = [0, 0, 1280, 1024];
        case 5
            scrSize = [0, 0, 800, 600];
        case 6
            scrSize = [0, 0, str2double(get(handles.scrRow,'String')),...
                str2double(get(handles.scrCol,'String'))];
    end
    scrNum = str2double(get(handles.scrNum,'String'));
    numTrial = str2double(get(handles.numTrials,'String'));
    timeSti = str2double(get(handles.timeSti, 'String'));
    timeRest = str2double(get(handles.timeRest, 'String'));
    freq = str2double(strsplit(get(handles.freq,'String'),','));
    btwBox = str2double(get(handles.btwBox,'String'));
    boxSize =str2double(get(handles.boxSize, 'String'));
catch
    set(handles.noti_text, 'String','Somethings Wrong...?');
end
if(~isequal(sock.status, 'closed'))
    if(~isequal(scrSize,[0 0 0 0]))
        s = Makeparadigm_SSVEP_manual({'time_sti',timeSti;'num_trials',numTrial;...
            'time_rest',timeRest;'freq',freq;'boxSize',boxSize;...
            'betweenBox',btwBox;'screen_size',scrSize; 'num_screen', scrNum});
    else
        s = Makeparadigm_SSVEP_manual({'time_sti',timeSti;'num_trials',numTrial;...
            'time_rest',timeRest;'freq',freq;'boxSize',boxSize;...
            'betweenBox',btwBox;'num_screen', scrNum});
    end
else
    set(handles.noti_text, 'String','Check your socket');
    return;
end
set(handles.noti_text, 'String',s);




% --- Executes on selection change in scrPop.
function scrPop_Callback(hObject, eventdata, handles)
% hObject    handle to scrPop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns scrPop contents as cell array
%        contents{get(hObject,'Value')} returns selected item from scrPop
if(get(handles.scrPop,'Value') == 6)
    scrPanelOn(handles, 'on');
else
    scrPanelOn(handles, 'off');
end

function scrPanelOn(handles, onoff)
set(handles.scrRow,'visible',onoff);
set(handles.scrCol,'visible',onoff);
set(handles.scrX,'visible',onoff);

% --- Executes during object creation, after setting all properties.
function scrPop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scrPop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over scrNum.
function scrNum_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to scrNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.scrNum,'String','');


% --- Executes on button press in server_button.
function server_button_Callback(hObject, eventdata, handles)
% hObject    handle to server_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock;
if ~isequal(get(handles.pport_check, 'String'), 'CHECK!')
    return;
end

if isempty(sock) || isequal(sock.status,'closed')
    port = str2double(get(handles.port,'String'));
    sock = tcpip('0.0.0.0', port, 'NetworkRole', 'server', 'timeout', 2);
    set(handles.noti_text, 'String','Waiting!');
    fopen(sock);
    
    for i = 1:5
        [trigger, a] = fread(sock,1);
        if(a > 0)
            break;
        end
    end
    ppTrigger(trigger);
    set(handles.noti_text, 'String','Connection');
else
    set(handles.noti_text, 'String','Hmmm.....');
end


function port_Callback(hObject, eventdata, handles)
% hObject    handle to port (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of port as text
%        str2double(get(hObject,'String')) returns contents of port as a double
input = get(handles.timeSti,'String');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String',sprintf('[%s] is not acceptable', input))
    set(handles.timeSti,'String','5');
end

% --- Executes during object creation, after setting all properties.
function port_CreateFcn(hObject, eventdata, handles)
% hObject    handle to port (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pport_Callback(hObject, eventdata, handles)
% hObject    handle to pport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pport as text
%        str2double(get(hObject,'String')) returns contents of pport as a double


% --- Executes during object creation, after setting all properties.
function pport_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function RESET(handles)
global sock
scrPanelOn(handles, 'off');
set(handles.scrPop,'Value',1);
set(handles.scrNum,'String', '2');
set(handles.numTrials,'String', '10');
set(handles.timeSti, 'String', '5');
set(handles.timeRest, 'String', '2');
set(handles.freq,'String', '6.67, 7.5, 8.57,10, 12');
set(handles.btwBox,'String', '200');
set(handles.boxSize, 'String', '150');
set(handles.pport, 'String', 'D010');
set(handles.port, 'String', '12300');
set(handles.server_button,'String','Open');
try
    fclose(sock);
catch
end
set(handles.noti_text, 'String','Welcome');


% --- Executes on button press in pport_check.
function pport_check_Callback(hObject, eventdata, handles)
% hObject    handle to pport_check (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global IO_ADDR IO_LIB
try
    % Trigger
    pport = get(handles.pport,'String');
    IO_ADDR=hex2dec(pport);
    IO_LIB=which('inpoutx64.dll');
catch
    set(handles.noti_text, 'String','Check your Parellal port');
    return;
end
set(handles.pport_check,'String','CHECK!');
