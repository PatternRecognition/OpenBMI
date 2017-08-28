function varargout = online_ssvep_gui_analysis(varargin)
% ONLINE_SSVEP_GUI_ANALYSIS MATLAB code for online_ssvep_gui_analysis.fig
%      ONLINE_SSVEP_GUI_ANALYSIS, by itself, creates a new ONLINE_SSVEP_GUI_ANALYSIS or raises the existing
%      singleton*.
%
%      H = ONLINE_SSVEP_GUI_ANALYSIS returns the handle to a new ONLINE_SSVEP_GUI_ANALYSIS or the handle to
%      the existing singleton*.
%
%      ONLINE_SSVEP_GUI_ANALYSIS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ONLINE_SSVEP_GUI_ANALYSIS.M with the given input arguments.
%
%      ONLINE_SSVEP_GUI_ANALYSIS('Property','Value',...) creates a new ONLINE_SSVEP_GUI_ANALYSIS or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before online_ssvep_gui_analysis_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to online_ssvep_gui_analysis_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help online_ssvep_gui_analysis

% Last Modified by GUIDE v2.5 14-Aug-2017 19:12:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @online_ssvep_gui_analysis_OpeningFcn, ...
    'gui_OutputFcn',  @online_ssvep_gui_analysis_OutputFcn, ...
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

% --- Executes just before online_ssvep_gui_analysis is made visible.
function online_ssvep_gui_analysis_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to online_ssvep_gui_analysis (see VARARGIN)

% Choose default command line output for online_ssvep_gui_analysis
% save('InitData.mat','handles');
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
initialize_gui(hObject, handles, false);

% UIWAIT makes online_ssvep_gui_analysis wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = online_ssvep_gui_analysis_OutputFcn(hObject, eventdata, handles)
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
RESET(handles, true);
% Update handles structure
guidata(handles.figure1, handles);


function freq_Callback(hObject, eventdata, handles)
% hObject    handle to freq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freq as text
%        str2double(get(hObject,'String')) returns contents of freq as a double
input = get(handles.freq,'string');
if regexp(input, '[^0-9., ;]')
    set(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
    set(handles.freq,'string','6.67, 7.5, 8.57,10, 12');
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



function time_stimulus_Callback(hObject, eventdata, handles)
% hObject    handle to time_stimulus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of time_stimulus as text
%        str2double(get(hObject,'String')) returns contents of time_stimulus as a double
input = get(handles.time_stimulus,'string');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
    set(handles.timeSti,'string','5');
end

% --- Executes during object creation, after setting all properties.
function time_stimulus_CreateFcn(hObject, eventdata, handles)
% hObject    handle to time_stimulus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function freq_range_min_Callback(hObject, eventdata, handles)
% hObject    handle to freq_range_min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freq_range_min as text
%        str2double(get(hObject,'String')) returns contents of freq_range_min as a double
input = get(handles.freq_range_min,'string');
if regexp(input, '[^0-9.]')
    set(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
    set(handles.freq_range_min,'string','0.1');
end

% --- Executes during object creation, after setting all properties.
function freq_range_min_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freq_range_min (see GCBO)
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
RESET(handles,false);

% --- Executes on button press in btnStart.
function btnStart_Callback(hObject, eventdata, handles)
% hObject    handle to btnStart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock state
try
    chan = get(handles.chan,'String');
    if isequal(chan,'all')
        chan = filt_EEG_CHANNEL(state.clab);
        chan = cell2mat(chan(1,:));
    else
        chan = str2double(strsplit(chan, ' ,'));
    end
    time_stimulus = str2double(get(handles.time_stimulus, 'String'));
    freq = str2double(strsplit(get(handles.freq, 'String'),','));
    fRange = [str2double(get(handles.freq_range_min, 'String')), ...
        str2double(get(handles.freq_range_max,'String'))];
    freq_class = strsplit(get(handles.freq_class,'String'),',');
    axes_fft = handles.axes_FFT;
    axes_cca = handles.axes_CCA;
    results_text = handles.results_text;
catch
    set(handles.noti_text, 'String', 'Somethings wrong...');
    return;
end
if ~isequal(sock.status,'closed')
    s = online_ssvep_Analysis(axes_fft, axes_cca, results_text, {'chan', chan;...
        'time_stimulus', time_stimulus; 'freq', freq;...
        'fRange', fRange; 'class', freq_class});
else
    set(handles.noti_text, 'String', 'check your socket');
    return;
end
bbci_acquire_bv('close');
set(handles.noti_text, 'String', s);


function chan_Callback(hObject, eventdata, handles)
% hObject    handle to chan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of chan as text
%        str2double(get(hObject,'String')) returns contents of chan as a double
input = get(handles.chan,'string');
if ~isequal(input,'all')
    if regexp(input, '[^0-9., ;]')
        set(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
        set(handles.freq,'string','all');
    end
end

% --- Executes during object creation, after setting all properties.
function chan_CreateFcn(hObject, eventdata, handles)
% hObject    handle to chan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in trig_button.
function trig_button_Callback(hObject, eventdata, handles)
% hObject    handle to trig_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock state

try
    bbci_acquire_bv('close');
    params = struct;
    state = bbci_acquire_bv('init', params);
catch
    set(handles.noti_text,'String', 'Please open the BrainVisino Recorder');
    return;
end

if isempty(sock) || isequal(sock.status,'closed')
    port = str2double(get(handles.tcpip_port,'String'));
    sock = tcpip('localhost', port,'timeout',2);
    try
        fopen(sock);
    catch
        set(handles.noti_text, 'String', 'First open the server');
        return;
    end
    while true
        fwrite(sock,19);
        [~, ~, markerdescr, ~] = bbci_acquire_bv(state);
        if ~isempty(markerdescr)&&(markerdescr == 19)
            flushoutput(sock);
            break;
        end
    end
    set(handles.closebtn,'Visible', 'On');
    set(handles.trig_button, 'String', 'OK!');
    set(handles.noti_text, 'String', 'Connect!');
end


% --- Executes on button press in selChan.
function selChan_Callback(hObject, eventdata, handles)
% hObject    handle to selChan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global state
filt_EEG_CHANNEL(state.clab);
select_channel_v2({'chan', state.clab});
% figure;
% uitable('Data',handles.initialization.bv.clab);



function freq_class_Callback(hObject, eventdata, handles)
% hObject    handle to freq_class (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freq_class as text
%        str2double(get(hObject,'String')) returns contents of freq_class as a double


% --- Executes during object creation, after setting all properties.
function freq_class_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freq_class (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tcpip_port_Callback(hObject, eventdata, handles)
% hObject    handle to tcpip_port (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tcpip_port as text
%        str2double(get(hObject,'String')) returns contents of tcpip_port as a double
input = get(handles.tcpip_port,'string');
if regexp(input, '[^0-9]')
    set(handles.noti_text, 'String', sprintf('[%s] is not acceptable', input))
    set(handles.timeSti,'string','12300');
end


% --- Executes during object creation, after setting all properties.
function tcpip_port_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tcpip_port (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function RESET(handles, init)
% handles    empty - handles not created until after all CreateFcns called
% init        initialization
set(handles.chan,'String','all');
set(handles.time_stimulus, 'String','5');
set(handles.freq, 'String','6.67, 7.5, 8.57,10, 12');
set(handles.freq_range_min, 'String', '0.1');
set(handles.freq_range_max, 'String', '50');
set(handles.freq_class,'String','up, left, center, right, down');
set(handles.trig_button,'String', 'Check');
set(handles.closebtn,'Visible', 'Off');
set(handles.results_text, 'String', '');
set(handles.noti_text, 'String', 'Welcome');
set(handles.axes_FFT, 'XTickLabel','');
set(handles.axes_CCA, 'XTickLabel', '');
set(handles.axes_FFT, 'YTickLabel', '');
set(handles.axes_CCA, 'YTickLabel', '');
delete(get(handles.axes_FFT,'Children'));
delete(get(handles.axes_CCA,'Children'));
global sock
try
    bbci_acquire_bv('close');
catch
    set(handles.noti_text,'String', 'Please open the BV');
end
try
    fclose(sock);
catch
end

function freq_range_max_Callback(hObject, eventdata, handles)
% hObject    handle to freq_range_max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freq_range_max as text
%        str2double(get(hObject,'String')) returns contents of freq_range_max as a double
input = get(handles.freq_range_max,'string');
if regexp(input, '[^0-9.]')
    set(handles.noti_text, 'String',  sprintf('[%s] is not acceptable', input))
    set(handles.timeSti,'string','50');
end


% --- Executes during object creation, after setting all properties.
function freq_range_max_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freq_range_max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in closebtn.
function closebtn_Callback(hObject, eventdata, handles)
% hObject    handle to closebtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock;
fclose(sock);
set(handles.noti_text, 'String', 'Close the socket');
set(handles.trig_button,'String', 'Check');
set(handles.closebtn,'Visible', 'Off');
