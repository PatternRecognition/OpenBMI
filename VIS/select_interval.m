function varargout = select_interval(varargin)
% SELECT_INTERVAL MATLAB code for select_interval.fig
%      SELECT_INTERVAL, by itself, creates a new SELECT_INTERVAL or raises the existing
%      singleton*.
%
%      H = SELECT_INTERVAL returns the handle to a new SELECT_INTERVAL or the handle to
%      the existing singleton*.
%
%      SELECT_INTERVAL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SELECT_INTERVAL.M with the given input arguments.
%
%      SELECT_INTERVAL('Property','Value',...) creates a new SELECT_INTERVAL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before select_interval_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to select_interval_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help select_interval

% Last Modified by GUIDE v2.5 10-Nov-2017 11:48:08

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @select_interval_OpeningFcn, ...
                   'gui_OutputFcn',  @select_interval_OutputFcn, ...
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


% --- Executes just before select_interval is made visible.
function select_interval_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to select_interval (see VARARGIN)

% Choose default command line output for select_interval
handles.output = hObject;

if isempty(varargin)
    handles.init_ival=[0,100;100,200;200,300;300,400;400,500];
else
    handles.init_ival=varargin{1};
end

% Update handles structure
guidata(hObject, handles);
initialize_gui(hObject, handles, false);

% UIWAIT makes select_interval wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = select_interval_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
delete(handles.figure1);


function initialize_gui(hObject, handles, isreset)
% If the metricdata field is present and the reset flag is false, it means
% we are we are just re-initializing a GUI by calling it from the cmd line
% while it is up. So, bail out as we dont want to reset the data.
% Update handles structure
% guidata(handles.figure1, handles);
RESET(hObject,handles);



function RESET(hObject,handles, init)
% handles    empty - handles not created until after all CreateFcns called
% init        initialization
ival=handles.init_ival;
[n m]=size(ival);
ival_name=[{'start'}, {'end'}];
for i=1:n
    for j=1:m
        eval(sprintf('handles.ival%d_%s.String=ival(i,j);',i,ival_name{j}));
    end
end
guidata(hObject,handles);

function UPDATE(hObject,handles,init)
% handles    structure with handles and user data (see GUIDATA)
% init        initialization
ival_name={'start', 'end'};
handles.selected_ival=[str2num(handles.ival1_start.String), str2num(handles.ival1_end.String)];
for i=2:5
    eval(sprintf('handles.selected_ival=[handles.selected_ival;str2num(handles.ival%d_%s.String),str2num(handles.ival%d_%s.String)];',i,ival_name{1},i,ival_name{2}));   
end

handles.output=handles.selected_ival;
guidata(hObject,handles);


% --- Executes on button press in apply_btn.
function apply_btn_Callback(hObject, eventdata, handles)
% hObject    handle to apply_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
UPDATE(hObject, handles,false);
uiresume(handles.figure1);
% The figure can be deleted now


% --- Executes on button press in reset_btn.
function reset_btn_Callback(hObject, eventdata, handles)
% hObject    handle to reset_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
RESET(handles,false);



function ival1_start_Callback(hObject, eventdata, handles)
% hObject    handle to ival1_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival1_start as text
%        str2double(get(hObject,'String')) returns contents of ival1_start as a double


% --- Executes during object creation, after setting all properties.
function ival1_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival1_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival1_end_Callback(hObject, eventdata, handles)
% hObject    handle to ival1_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival1_end as text
%        str2double(get(hObject,'String')) returns contents of ival1_end as a double


% --- Executes during object creation, after setting all properties.
function ival1_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival1_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival2_start_Callback(hObject, eventdata, handles)
% hObject    handle to ival2_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival2_start as text
%        str2double(get(hObject,'String')) returns contents of ival2_start as a double


% --- Executes during object creation, after setting all properties.
function ival2_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival2_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival2_end_Callback(hObject, eventdata, handles)
% hObject    handle to ival2_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival2_end as text
%        str2double(get(hObject,'String')) returns contents of ival2_end as a double


% --- Executes during object creation, after setting all properties.
function ival2_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival2_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival3_start_Callback(hObject, eventdata, handles)
% hObject    handle to ival3_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival3_start as text
%        str2double(get(hObject,'String')) returns contents of ival3_start as a double


% --- Executes during object creation, after setting all properties.
function ival3_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival3_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival3_end_Callback(hObject, eventdata, handles)
% hObject    handle to ival3_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival3_end as text
%        str2double(get(hObject,'String')) returns contents of ival3_end as a double


% --- Executes during object creation, after setting all properties.
function ival3_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival3_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival4_start_Callback(hObject, eventdata, handles)
% hObject    handle to ival4_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival4_start as text
%        str2double(get(hObject,'String')) returns contents of ival4_start as a double


% --- Executes during object creation, after setting all properties.
function ival4_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival4_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival4_end_Callback(hObject, eventdata, handles)
% hObject    handle to ival4_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival4_end as text
%        str2double(get(hObject,'String')) returns contents of ival4_end as a double


% --- Executes during object creation, after setting all properties.
function ival4_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival4_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival5_start_Callback(hObject, eventdata, handles)
% hObject    handle to ival5_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival5_start as text
%        str2double(get(hObject,'String')) returns contents of ival5_start as a double


% --- Executes during object creation, after setting all properties.
function ival5_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival5_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ival5_end_Callback(hObject, eventdata, handles)
% hObject    handle to ival5_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ival5_end as text
%        str2double(get(hObject,'String')) returns contents of ival5_end as a double


% --- Executes during object creation, after setting all properties.
function ival5_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival5_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
