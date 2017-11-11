function varargout = plot_controller(varargin)
% PLOT_CONTROLLER MATLAB code for plot_controller.fig
%      PLOT_CONTROLLER, by itself, creates a new PLOT_CONTROLLER or raises the existing
%      singleton*.
%       
%       excute with plot_controller(SMT)
%      H = PLOT_CONTROLLER returns the handle to a new PLOT_CONTROLLER or the handle to
%      the existing singleton*.
%
%      PLOT_CONTROLLER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLOT_CONTROLLER.M with the given input arguments.
%
%      PLOT_CONTROLLER('Property','Value',...) creates a new PLOT_CONTROLLER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before plot_controller_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to plot_controller_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help plot_controller

% Last Modified by GUIDE v2.5 10-Nov-2017 20:54:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @plot_controller_OpeningFcn, ...
                   'gui_OutputFcn',  @plot_controller_OutputFcn, ...
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


% --- Executes just before plot_controller is made visible.
function plot_controller_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to plot_controller (see VARARGIN)

% Choose default command line output for plot_controller
handles.output = hObject;
if ~sum(size(varargin))
    warning('OpenBMI: No data input');return
elseif ~isfield(varargin{1},'x')
    warning('OpenBMI: Data must have fields named ''x''');return
else
    handles.data=varargin{1};
end
% Update handles structure
guidata(hObject, handles);
initialize_gui(hObject, handles, false);
% UIWAIT makes plot_controller wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = plot_controller_OutputFcn(hObject, eventdata, handles) 
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
set(handles.sampling_rate,'String',handles.data.fs);
set(handles.chan_num,'String',length(handles.data.chan));
set(handles.time_seg_start, 'String', handles.data.ival(1));
set(handles.time_seg_end, 'String', handles.data.ival(end));
chan=handles.data.class{1,2};
for i=2:size(handles.data.class(:,2))
    chan=strcat(chan,',',handles.data.class{i,2});
end
set(handles.class,'String',chan);
RESET(handles, true);
% Update handles structure
guidata(handles.figure1, handles);



function RESET(handles, init)
% handles    empty - handles not created until after all CreateFcns called
% init        initialization
set(handles.check_time_plot,'Value',true);
set(handles.check_topography,'Value',true);
set(handles.chan_listbox, 'String', sprintf('Cz\nOz'));
% set(handles.chan_listbox, 'Value', []);
% set(handles.baseline_start, 'String', -100);
% set(handles.baseline_end, 'String', 0);


chan=handles.data.class{1,2};
for i=2:size(handles.data.class(:,2))
    chan=strcat(chan,'\n',handles.data.class{i,2});
end
set(handles.class_listbox,'String',chan);

ival=[0,100;100,200;200,300;300,400;400,500];
% show_ival='';
% for i=1:length(ival)
%     show_ival=strcat(show_ival,sprintf('%d ~ %d\n', ival(i,1), ival(i,2)));
% end
% set(handles.list_ival, 'String', show_ival);
set(handles.ival_listbox, 'String', sprintf('%d ~ %d\n%d ~ %d\n%d ~ %d\n%d ~ %d\n%d ~ %d',ival(1,1),ival(1,2),ival(2,1),ival(2,2),ival(3,1),ival(3,2),ival(4,1),ival(4,2),ival(5,1),ival(5,2)));


% --- Executes on button press in load_data_btn.
function load_data_btn_Callback(hObject, eventdata, handles)
% hObject    handle to load_data_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in select_chan_btn.
function select_chan_btn_Callback(hObject, eventdata, handles)
% hObject    handle to select_chan_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
select_channel_v2({'chan', {'Fpz', ...
       'Fp1','AFp1','AFp2','Fp2', ...
       'AF7','AF5','AF3','AFz','AF4','AF6','AF8', ...
       'FAF5','FAF1','FAF2'}});

% --- Executes on button press in select_ival_btn.
function select_ival_btn_Callback(hObject, eventdata, handles)
% hObject    handle to select_ival_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
select_interval;

% --- Executes on selection change in chan_listbox.
function chan_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to chan_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns chan_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from chan_listbox


% --- Executes during object creation, after setting all properties.
function chan_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to chan_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_listbox.
function ival_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to ival_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns ival_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_listbox


% --- Executes during object creation, after setting all properties.
function ival_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in check_time_plot.
function check_time_plot_Callback(hObject, eventdata, handles)
% hObject    handle to check_time_plot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_time_plot


% --- Executes on button press in check_topography.
function check_topography_Callback(hObject, eventdata, handles)
% hObject    handle to check_topography (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_topography


% --- Executes on button press in draw_btn.
function draw_btn_Callback(hObject, eventdata, handles)
% hObject    handle to draw_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure;


% --- Executes on button press in reset_btn.
function reset_btn_Callback(hObject, eventdata, handles)
% hObject    handle to reset_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
RESET(handles,false);



function time_seg_start_Callback(hObject, eventdata, handles)
% hObject    handle to time_seg_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of time_seg_start as text
%        str2double(get(hObject,'String')) returns contents of time_seg_start as a double


% --- Executes during object creation, after setting all properties.
function time_seg_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to time_seg_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function time_seg_end_Callback(hObject, eventdata, handles)
% hObject    handle to time_seg_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of time_seg_end as text
%        str2double(get(hObject,'String')) returns contents of time_seg_end as a double


% --- Executes during object creation, after setting all properties.
function time_seg_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to time_seg_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function freq_band_start_Callback(hObject, eventdata, handles)
% hObject    handle to freq_band_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freq_band_start as text
%        str2double(get(hObject,'String')) returns contents of freq_band_start as a double


% --- Executes during object creation, after setting all properties.
function freq_band_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freq_band_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function freq_band_end_Callback(hObject, eventdata, handles)
% hObject    handle to freq_band_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of freq_band_end as text
%        str2double(get(hObject,'String')) returns contents of freq_band_end as a double


% --- Executes during object creation, after setting all properties.
function freq_band_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to freq_band_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function baseline_start_Callback(hObject, eventdata, handles)
% hObject    handle to baseline_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baseline_start as text
%        str2double(get(hObject,'String')) returns contents of baseline_start as a double


% --- Executes during object creation, after setting all properties.
function baseline_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baseline_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function baseline_end_Callback(hObject, eventdata, handles)
% hObject    handle to baseline_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baseline_end as text
%        str2double(get(hObject,'String')) returns contents of baseline_end as a double


% --- Executes during object creation, after setting all properties.
function baseline_end_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baseline_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
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


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in select_class_btn.
function select_class_btn_Callback(hObject, eventdata, handles)
% hObject    handle to select_class_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in class_listbox.
function class_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to class_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns class_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from class_listbox


% --- Executes during object creation, after setting all properties.
function class_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to class_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
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
