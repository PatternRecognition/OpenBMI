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

% Last Modified by GUIDE v2.5 11-Apr-2018 18:59:36

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
    error('OpenBMI: No data input');
elseif ~isfield(varargin{1},'x')
    error('OpenBMI: Data must have fields named ''x''');
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

function initialize_gui(hObject, handles, isreset)
% If the metricdata field is present and the reset flag is false, it means
% we are we are just re-initializing a GUI by calling it from the cmd line
% while it is up. So, bail out as we dont want to reset the data.
% set(handles.time_seg_start, 'String', handles.data.ival(1));
% set(handles.time_seg_end, 'String', handles.data.ival(end));
set(handles.time_seg, 'String', sprintf('%d ~ %d', handles.data.ival(1), handles.data.ival(end)));
set(handles.sampling_rate,'String',handles.data.fs);
set(handles.num_class,'String',size(handles.data.class, 1));
set(handles.chan_num,'String',length(handles.data.chan));

chan=handles.data.chan(1);
for i=2:length(handles.data.chan)
    chan=strcat(chan,{', '},handles.data.chan(i));
end
set(handles.chan_info,'String',chan);

% Update handles structure
guidata(hObject, handles);
RESET(hObject,handles);




function RESET(hObject, handles, isreset)
% handles     structure with handles and user data (see GUIDATA)
% init        initialization
% Initialize plot type
set(handles.check_ERP,'Value',false);
set(handles.check_SSVEP, 'Value', false);
set(handles.check_MI,'Value',false);
set(handles.check_Topo,'Value',false);

% Initialize channel
handles.selected_chan={'Cz','Oz'};
str={};
for i=1:length(handles.selected_chan)
    str=[str; sprintf('%s',handles.selected_chan{i})];
end
% sprintf('%s',selected_chan{1});
set(handles.chan_listbox, 'String', str);

% Initialize interval
str ={};

ival = handles.data.ival([1, end]);
if ival(1) < 0, ival(1) = 0; end
selected_ival = linspace(ival(1), ival(2), 6);
handles.selected_ival=[selected_ival(1),selected_ival(2);...
    selected_ival(2),selected_ival(3);selected_ival(3),selected_ival(4);...
    selected_ival(4),selected_ival(5); selected_ival(5),selected_ival(6)];
ival=handles.selected_ival;
for i = 1:size(ival,1)
    str = [str; sprintf('%d ~ %d',ival(i,1), ival(i,2))];
end
set(handles.ival_listbox, 'String', str);

% Initialize num_class
str = {};
handles.selected_class=handles.data.class(:,2);

for i=1:length(handles.selected_class)
    str = [str; sprintf('%s', handles.selected_class{i})];
end

set(handles.inputToggle, 'Value', false);
set(handles.inputToggle, 'String', 'Direct');
set(handles.pop_range, 'Visible', 'on');
set(handles.maxTopo, 'Visible', 'off');
set(handles.minTopo, 'Visible', 'off');
set(handles.maxTopo, 'String', '0');
set(handles.minTopo, 'String', '0');
set(handles.tildeTopo, 'Visible', 'off');
set(handles.selToggle, 'Value', false);
set(handles.selToggle, 'String', 'Select');
set(handles.maxSelect, 'Enable', 'off');%
set(handles.minSelect, 'Enable', 'off');
set(handles.maxSelect, 'String', '0');
set(handles.minSelect, 'String', '0');
set(handles.ylimToggle, 'Value', false);
set(handles.ylimToggle, 'String', 'Select');
set(handles.maxYlim, 'Enable', 'off');
set(handles.minYlim, 'Enable', 'off');
set(handles.maxYlim, 'String', '0');
set(handles.minYlim, 'String', '0');

set(handles.class_listbox,'String', str);

set(handles.pop_range, 'Value', 1);
set(handles.pop_color, 'Value', 1);
set(handles.pop_quality, 'Value', 1);
set(handles.pop_orient, 'Value', 1);    

% Initialize baseline
set(handles.baseline_start, 'String', string(handles.data.ival(1)));
set(handles.baseline_end, 'String', '0');

set(handles.note_txt,'String', {'';'';'Welcome'});

% Update handles structure
guidata(hObject, handles);


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
try
    [~,selected_chan]=GUI_selectChannels(handles.data.chan,handles.selected_chan);
    if length(selected_chan) > 5
        set(handles.note_txt, 'String', {'';'';'Don''t you think that selected channels are too many?'});
        return;
    else
        handles.selected_chan = selected_chan;
    end
catch
    return;
end
str={};
for i=1:length(handles.selected_chan)
    str=[str; sprintf('%s',handles.selected_chan{i})];
end
set(handles.chan_listbox, 'String', str);
set(handles.note_txt, 'String', {'';'';'Channels are selected'});

clear tmp;
% Update handles structure
guidata(hObject, handles);


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


% --- Executes on button press in select_ival_btn.
function select_ival_btn_Callback(hObject, eventdata, handles)
% hObject    handle to select_ival_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
    ival = select_interval(handles.selected_ival);
    if ~isempty(ival)&&(min(ival(:)) < handles.data.ival(1) || max(ival(:)) > handles.data.ival(end))
        set(handles.note_txt,'String', {'';'';'Please select intervals between segmented data'});
        return;
    end
    if ~isempty(ival)
        ival = sort(ival,2);
        [~, sort_] = sort(ival,1);
        ival = ival(sort_(:,1),:);
    else
        ival = handles.data.ival([1, end]);
%         if ival(1) < 0, ival(1) = 0; end % ival 0 to max -> min to max
    end
catch
    return;
end
handles.selected_ival=ival;
% str=sprintf('%d ~ %d',handles.selected_ival(1,1), handles.selected_ival(1,2));
% for i=2:length(handles.selected_ival(:,1))
%     str=sprintf('%s\n%d ~ %d',str,handles.selected_ival(i,1), handles.selected_ival(i,2));
% end
str = {};
for i = 1:size(ival,1)
    str = [str; sprintf('%d ~ %d', ival(i,1), ival(i,2))];
end

set(handles.ival_listbox, 'String',str);
set(handles.note_txt, 'String', {'';'';'Itervals are selected'});

% Update handles structure
guidata(hObject, handles);


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

% --- Executes on button press in select_class_btn.
function select_class_btn_Callback(hObject, eventdata, handles)
% hObject    handle to select_class_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
    handles.selected_class=select_class(handles.data.class(:,2),handles.selected_class);
catch
    return;
end
str={};
for i=1:length(handles.selected_class)
    str=[str; sprintf('%s',handles.selected_class{i});];
end
set(handles.class_listbox, 'String', str);
clear tmp;
set(handles.note_txt, 'String', {'';'';'Classes are selected'});

% Update handles structure
guidata(hObject, handles);

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


% --- Executes on button press in check_ERP.
function check_ERP_Callback(hObject, eventdata, handles)
% hObject    handle to check_ERP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_ERP


% --- Executes on button press in check_MI.
function check_MI_Callback(hObject, eventdata, handles)
% hObject    handle to check_MI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_MI


% --- Executes on button press in draw_btn.
function draw_btn_Callback(hObject, eventdata, handles)
% hObject    handle to draw_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Update baseline range
baseline=[str2double(get(handles.baseline_start,'String')), str2double(get(handles.baseline_end,'String'))];
if baseline(1) > baseline(2)
    baseline = flip(baseline);
    set(handles.baseline_start,'String', baseline(1));
    set(handles.baseline_end, 'String', baseline(2));
end
%% Select time
if get(handles.selToggle, 'Value')
    selTime = [str2double(get(handles.minSelect, 'String')), str2double(get(handles.maxSelect, 'String'))];
    if selTime(1) > selTime(2)
        selTime = flip(selTime);
        set(handles.minSelect, 'String', selTime(1));
        set(handles.maxSelect, 'String', selTime(2));
    end
else
    selTime = [handles.data.ival(1), handles.data.ival(end)];
end
%% Ylim range
if get(handles.ylimToggle, 'Value')
    ylimRange = [str2double(get(handles.minYlim, 'String')), str2double(get(handles.maxYlim, 'String'))];
else
    ylimRange = [];
end
%% Get plots
if get(handles.check_ERP,'Value'), ERPPlot = 'on'; else ERPPlot = 'off'; end
if get(handles.check_SSVEP,'Value'), SSVEPPlot = 'on'; else SSVEPPlot = 'off'; end
if get(handles.check_MI,'Value'), MIPlot = 'on'; else MIPlot = 'off'; end
if get(handles.check_Topo,'Value'), TopoPlot = 'on'; else TopoPlot = 'off'; end
if ~sum(ismember({ERPPlot, MIPlot, SSVEPPlot, TopoPlot}, 'on'))
    set(handles.note_txt, 'String', {'';'';'Choose at least one plot type'});
    return;
end
%% Patch
if get(handles.check_patch,'Value'), Patch = 'on'; else Patch = 'off'; end
%% Topography range
if get(handles.inputToggle, 'Value')
    p_range = [str2double(get(handles.minTopo, 'String')), str2double(get(handles.maxTopo, 'String'))];
    if p_range(1) > p_range(2)
        p_range = flip(p_range);
        set(handles.minTopo,'String', p_range(1));
        set(handles.maxTopo,'String', p_range(2));
    end
else
    switch get(handles.pop_range, 'Value')
        case 1
            p_range = 'sym';
        case 2
            p_range = '0tomax';
        case 3
            p_range = 'minto0';
        case 4
            p_range = 'mintomax';
        case 5
            p_range = 'mean';
    end
end
%% Topography color
switch get(handles.pop_color, 'Value')
    case 1
        cm = 'parula';
    case 2
        cm = 'jet';
    case 3
        cm = 'hsv';
end
%% Topography quality
switch get(handles.pop_quality, 'Value')
    case 1
        quality = 'high';
    case 2
        quality = 'medium';
    case 3
        quality = 'low';
end

%% Interval check
if sum(handles.selected_ival(:,1) < selTime(1)) || sum(handles.selected_ival(:,2) > selTime(2))
    set(handles.note_txt, 'String', {'';'';'Choose intervals between selected time'});
    return;
end

%% Start visualization
set(handles.note_txt, 'String', {'';'';'Wait for Drawing'}); drawnow;
try
    options = {'Interval', handles.selected_ival;...
        'Channels',handles.selected_chan;'Class',handles.selected_class;...
        'TimePlot', ERPPlot; 'TopoPlot', TopoPlot; 'ErdPlot', MIPlot;...
         'Range', p_range; 'baseline', baseline;...
        'Colormap', cm; 'Patch', Patch; 'Quality', quality; ...
        'SelectTime', selTime; 'TimeRange', ylimRange};
    [avSMT, rSMT] = untitled_function(handles.data);
    output = vis_plotController(avSMT, rSMT, options);
catch error
    close gcf;
    output = {'';'Unexpected Error Occurred in';...
        sprintf('%s (line: %d)', error.stack(1).name, error.stack(1).line);...
        error.message};
end
set(handles.note_txt, 'String', output);
% visual_scalpPlot_fin(handles.data, {'Interval', handles.selected_ival;'Channels',{'Cz', 'POz','Oz'};'num_class',{'target','non-target'}});
% visual_scalpPlot_fin(handles.data, {'Interval', [-100 0 150 250 400];'Channels',{'Cz', 'POz','Oz'}});
% Update handles structure
% guidata(hObject, handles);

% --- Executes on button press in reset_btn.
function reset_btn_Callback(hObject, eventdata, handles)
% hObject    handle to reset_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
RESET(hObject,handles,false);



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
input = get(handles.baseline_start, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.baseline_start, 'String', string(handles.data.ival(1)));
    return;
end
if str2double(input) < handles.data.ival(1) || str2double(input) > handles.data.ival(end)
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input between the segmentation range'});
    set(handles.baseline_start, 'String', string(handles.data.ival(1)));
    return;
end


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
input = get(handles.baseline_end, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.baseline_end, 'String', '0');
    return;
end
if str2double(input) < handles.data.ival(1) || str2double(input) > handles.data.ival(end)
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input between the segmentation range'});
    set(handles.baseline_end, 'String', string(handles.data.ival(end)));
    return;
end


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


% --- Executes on button press in pushbutton6. % baseline
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to baseline_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baseline_start as text
%        str2double(get(hObject,'String')) returns contents of baseline_start as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baseline_start (see GCBO)
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


function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to baseline_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baseline_end as text
%        str2double(get(hObject,'String')) returns contents of baseline_end as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baseline_end (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in check_SSVEP.
function check_SSVEP_Callback(hObject, eventdata, handles)
% hObject    handle to check_SSVEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_SSVEP

% --- Executes on button press in check_Topo.
function check_Topo_Callback(hObject, eventdata, handles)
% hObject    handle to check_Topo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_Topo



% --- Executes on selection change in pop_range.
function pop_range_Callback(hObject, eventdata, handles)
% hObject    handle to pop_range (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pop_range contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pop_range


% --- Executes during object creation, after setting all properties.
function pop_range_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pop_range (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu10.
function popupmenu10_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu10 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu10


% --- Executes during object creation, after setting all properties.
function popupmenu10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu11.
function popupmenu11_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu11 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu11


% --- Executes during object creation, after setting all properties.
function popupmenu11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in pop_color.
function pop_color_Callback(hObject, eventdata, handles)
% hObject    handle to pop_color (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pop_color contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pop_color


% --- Executes during object creation, after setting all properties.
function pop_color_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pop_color (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in check_patch.
function check_patch_Callback(hObject, eventdata, handles)
% hObject    handle to check_patch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_patch


% --- Executes on selection change in pop_quality.
function pop_quality_Callback(hObject, eventdata, handles)
% hObject    handle to pop_quality (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pop_quality contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pop_quality


% --- Executes during object creation, after setting all properties.
function pop_quality_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pop_quality (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function minYlim_Callback(hObject, eventdata, handles)
% hObject    handle to minYlim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minYlim as text
%        str2double(get(hObject,'String')) returns contents of minYlim as a double
input = get(handles.minYlim, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.minYlim, 'String', '0');
    return;
end
% --- Executes during object creation, after setting all properties.
function minYlim_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minYlim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxYlim_Callback(hObject, eventdata, handles)
% hObject    handle to maxYlim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxYlim as text
%        str2double(get(hObject,'String')) returns contents of maxYlim as a double
input = get(handles.maxYlim, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.maxYlim, 'String', '0');
    return;
end

% --- Executes during object creation, after setting all properties.
function maxYlim_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxYlim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in inputToggle.
function inputToggle_Callback(hObject, eventdata, handles)
% hObject    handle to inputToggle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of inputToggle
if get(handles.inputToggle, 'Value')
    set(handles.pop_range, 'Visible', 'off'); 
    set(handles.maxTopo, 'Visible', 'on');
    set(handles.minTopo, 'Visible', 'on');
    set(handles.tildeTopo, 'Visible', 'on');
    set(handles.inputToggle, 'String', 'Options');
else
    set(handles.pop_range, 'Visible', 'on');
    set(handles.maxTopo, 'Visible', 'off');
    set(handles.minTopo, 'Visible', 'off');
    set(handles.tildeTopo, 'Visible', 'off');
    set(handles.inputToggle, 'String', 'Direct');
end



function minTopo_Callback(hObject, eventdata, handles)
% hObject    handle to minTopo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minTopo as text
%        str2double(get(hObject,'String')) returns contents of minTopo as a double
input = get(handles.minTopo, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.minTopo, 'String', '0');
    return;
end


% --- Executes during object creation, after setting all properties.
function minTopo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minTopo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxTopo_Callback(hObject, eventdata, handles)
% hObject    handle to maxTopo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxTopo as text
%        str2double(get(hObject,'String')) returns contents of maxTopo as a double
input = get(handles.maxTopo, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.maxTopo, 'String', '0');
    return;
end



% --- Executes during object creation, after setting all properties.
function maxTopo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxTopo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function minSelect_Callback(hObject, eventdata, handles)
% hObject    handle to minSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minSelect as text
%        str2double(get(hObject,'String')) returns contents of minSelect as a double
input = get(handles.minSelect, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.minSelect, 'String', string(handles.data.ival(1)));
    return;
end
if str2double(input) < handles.data.ival(1) || str2double(input) > handles.data.ival(end)
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input between the segmentation range'});
    set(handles.minSelect, 'String', string(handles.data.ival(1)));
    return;
end

% --- Executes during object creation, after setting all properties.
function minSelect_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxSelect_Callback(hObject, eventdata, handles)
% hObject    handle to maxSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxSelect as text
%        str2double(get(hObject,'String')) returns contents of maxSelect as a double
input = get(handles.maxSelect, 'String');
[~, len] = regexp(input, '^-?[0-9]+');

if ~isequal(len, length(input))
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input the number'});
    set(handles.maxSelect, 'String', string(handles.data.ival(end)));
    return;
end
if str2double(input) < handles.data.ival(1) || str2double(input) > handles.data.ival(end)
    set(handles.note_txt, 'String',{'';sprintf('[%s] is not acceptable', input);'please input between the segmentation range'});
    set(handles.maxSelect, 'String', string(handles.data.ival(end)));
    return;
end

% --- Executes during object creation, after setting all properties.
function maxSelect_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in selToggle.
function selToggle_Callback(hObject, eventdata, handles)
% hObject    handle to selToggle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of selToggle
if get(handles.selToggle, 'Value')
    set(handles.maxSelect, 'Enable', 'on');
    set(handles.minSelect, 'Enable', 'on');
    set(handles.selToggle, 'String', 'Deselect');
else
    set(handles.maxSelect, 'Enable', 'off');
    set(handles.minSelect, 'Enable', 'off');
	set(handles.selToggle, 'String', 'Select');
end

% --- Executes on button press in ylimToggle.
function ylimToggle_Callback(hObject, eventdata, handles)
% hObject    handle to ylimToggle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ylimToggle
if isequal(get(handles.ylimToggle, 'String'), 'Select')
    set(handles.maxYlim, 'Enable', 'on');
    set(handles.minYlim, 'Enable', 'on');
    set(handles.ylimToggle, 'String', 'Deselect');
    set(handles.ylimToggle, 'Value', true);
else
    set(handles.maxYlim, 'Enable', 'off');
    set(handles.minYlim, 'Enable', 'off');
    set(handles.ylimToggle, 'String', 'Select');
    set(handles.ylimToggle, 'Value', false);
end


% --- Executes on selection change in pop_orient.
function pop_orient_Callback(hObject, eventdata, handles)
% hObject    handle to pop_orient (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pop_orient contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pop_orient


% --- Executes during object creation, after setting all properties.
function pop_orient_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pop_orient (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
