function varargout = p300_gui(varargin)
% P300_GUI M-file for p300_gui.fig
%      P300_GUI, by itself, creates a new P300_GUI or raises the existing
%      singleton*.
%
%      H = P300_GUI returns the handle to a new P300_GUI or the handle to
%      the existing singleton*.
%
%      P300_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in P300_GUI.M with the given input arguments.
%
%      P300_GUI('Property','Value',...) creates a new P300_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before p300_gui_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to p300_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help p300_gui

% Last Modified by GUIDE v2.5 03-Nov-2010 11:00:24

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @p300_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @p300_gui_OutputFcn, ...
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


% --- Executes just before p300_gui is made visible.
function p300_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to p300_gui (see VARARGIN)

% Choose default command line output for p300_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes p300_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);
addpath(strcat(fileparts(mfilename('fullpath')), '/setups/'));


% --- Outputs from this function are returned to the command line.
function varargout = p300_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in ival_start_1.
function ival_start_1_Callback(hObject, eventdata, handles)
% hObject    handle to ival_start_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_start_1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_start_1
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_start_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_start_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_end_1.
function ival_end_1_Callback(hObject, eventdata, handles)
% hObject    handle to ival_end_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_end_1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_end_1
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_end_1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_end_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_start_2.
function ival_start_2_Callback(hObject, eventdata, handles)
% hObject    handle to ival_start_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_start_2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_start_2
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_start_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_start_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_end_2.
function ival_end_2_Callback(hObject, eventdata, handles)
% hObject    handle to ival_end_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_end_2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_end_2
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_end_2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_end_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_start_3.
function ival_start_3_Callback(hObject, eventdata, handles)
% hObject    handle to ival_start_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_start_3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_start_3
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_start_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_start_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_end_3.
function ival_end_3_Callback(hObject, eventdata, handles)
% hObject    handle to ival_end_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_end_3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_end_3
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_end_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_end_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_start_4.
function ival_start_4_Callback(hObject, eventdata, handles)
% hObject    handle to ival_start_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_start_4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_start_4
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_start_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_start_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_end_4.
function ival_end_4_Callback(hObject, eventdata, handles)
% hObject    handle to ival_end_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_end_4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_end_4
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_end_4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_end_4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_start_5.
function ival_start_5_Callback(hObject, eventdata, handles)
% hObject    handle to ival_start_5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_start_5 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_start_5
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_start_5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_start_5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ival_end_5.
function ival_end_5_Callback(hObject, eventdata, handles)
% hObject    handle to ival_end_5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ival_end_5 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ival_end_5
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
end

% --- Executes during object creation, after setting all properties.
function ival_end_5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ival_end_5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in use_ivals_button.
function use_ivals_button_Callback(hObject, eventdata, handles)
% hObject    handle to use_ivals_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    redraw_intervals(handles);
    ivals = get_ivals(handles, 1);
    set_ivals(handles,sort(ivals,1));
    transfer_ivals_to_base(handles);
else
    add_to_message_box(handles, 'Please initialize first and load data...');
end

function message_box_Callback(hObject, eventdata, handles)
% hObject    handle to message_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of message_box as text
%        str2double(get(hObject,'String')) returns contents of message_box as a double


% --- Executes during object creation, after setting all properties.
function message_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to message_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in file_list_box.
function file_list_box_Callback(hObject, eventdata, handles)
% hObject    handle to file_list_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns file_list_box contents as cell array
%        contents{get(hObject,'Value')} returns selected item from file_list_box


% --- Executes during object creation, after setting all properties.
function file_list_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to file_list_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in load_data_button.
function load_data_button_Callback(hObject, eventdata, handles)
% hObject    handle to load_data_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
add_to_message_box(handles, 'Loading data. Please wait.....');drawnow;
sel_files = get(handles.file_list_box, 'string');
evalin('base', 'clear Cnt epo mrk mnt epo_r');
assignin('base', 'tmp', sel_files(get(handles.file_list_box, 'value')));
evalin('base', 'bbci.train_file = tmp;clear tmp');
assignin('base', 'do_reject_channels', get(handles.reject_channels_tick, 'value'));
if check_if_initialized(handles),
    try
        set_button_state(handles, 'off', {''});
        evalin('base', ['bbci_bet_prepare; p300gui_preprocess_' get(handles.save_study, 'string')]);
        rtrials = evalin('base', 'rtrials');
        eyetrials = evalin('base', 'iArte');
        tottrials = evalin('base', 'mrk.toe');
        rclab = evalin('base', 'rclab');
        set_button_state(handles, 'on', {''});
        add_to_message_box(handles, 'Data loaded.');
        add_to_message_box(handles, ['Warning: ' num2str(length(rtrials)+length(eyetrials)) ' (' num2str(round((length(rtrials)+length(eyetrials))*100/length(tottrials))) '% of ' num2str(length(tottrials)) ') trials disregarded']);
        if ~isempty(rclab),
            add_to_message_box(handles, sprintf('Warning: channels %s removed.', sprintf('%s , ', rclab{:})));
        end
        update_image_axis(handles);
        reset_ivals_from_base(handles);
        reset_channel_names(handles);
        update_nr_chan_selected(handles);
    catch
        error = lasterror;
        disp(error.message);        
        set_button_state(handles, 'on', {''});
        add_to_message_box(handles, 'Something went wrong, so I didn''t load any data. Please check the command window for detailed errors');
    end
else
    add_to_message_box(handles, 'Please initialize first...');    
end



% --- Executes on button press in init_button.
function init_button_Callback(hObject, eventdata, handles)
% hObject    handle to init_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
     if strcmp(get(handles.vp_code_box, 'string'), 'Enter VP_CODE'),
         add_to_message_box(handles, 'Error: No VP_CODE set.');
     elseif isempty(strmatch('VP', get(handles.vp_code_box, 'string'))) && ~isempty(get(handles.vp_code_box, 'string')),
         add_to_message_box(handles, ...
             'Error: Invalid VP_CODE set. VP_CODE should start with VP');       
     else
         global TODAY_DIR VP_CODE
         if isempty(get(handles.vp_code_box, 'string')),
             evalin('base', 'clear VP_CODE; acq_makeDataFolder(''log_dir'',1)');
             set(handles.vp_code_box, 'string', VP_CODE);
         else
            VP_CODE = get(handles.vp_code_box, 'string');
         end
         studies = get(handles.study_box, 'string');
         evalin('base', 'clear Cnt epo epo_r opt');
         reset_axes(handles);
         text(1,  .7, 'Load data --> ', 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10);
         reset_ivals(handles);
         evalin('base',['p300gui_setup_online_' studies{get(handles.study_box,'value')}]);
         assignin('base', 'do_action', 'give_available_experiments');
         evalin('base', strcat('p300gui_runscripts_', studies{get(handles.study_box,'value')}, ';'));
         runs_available = evalin('base', 'available_experiments;');
         set(handles.experiment_box, 'string', runs_available);         
         look_for_existing_files(handles);
         look_for_existing_classifiers(handles);
         handle_experiment_parameters(handles, 'init');
         set(handles.save_study, 'string', studies{get(handles.study_box,'value')});
     end

function handle_experiment_parameters(handles, action, varargin);
        persistent params
        
        switch action
            case 'init'
                params_in = evalin('base', 'available_parameters');
                if iscell(params_in{1}), 
                    for i = 1:length(params_in), 
                        params{i} = propertylist2struct(params_in{i}{:});
                    end
                end
                handle_experiment_parameters(handles, 'update_exp');
                
            case 'update'
                exp_id = get(handles.experiment_box, 'value');
                param_name = get(handles.parameter_name_list, 'string');
                param_id = get(handles.parameter_name_list, 'value');
                set(handles.parameter_value_box, 'string', params{exp_id}.(param_name{param_id}));

            case 'save'
                exp_id = get(handles.experiment_box, 'value');
                param_name = get(handles.parameter_name_list, 'string');
                param_id = get(handles.parameter_name_list, 'value');
                value = get(handles.parameter_value_box, 'string');
                if isempty(str2num(value)),
                    params{exp_id}.(param_name{param_id}) = value;
                else
                    params{exp_id}.(param_name{param_id}) = str2num(value);
                end
                
            case 'update_exp'
                exp_id = get(handles.experiment_box, 'value');
                if ~isempty(params{exp_id}), 
                    set(handles.use_parameter_button, 'Visible', 'on');
                    set(handles.parameter_value_box, 'Visible', 'on');
                    set(handles.parameter_name_list, 'string', fieldnames(params{exp_id}), 'value', 1, 'Visible', 'on');
                    handle_experiment_parameters(handles, 'update');
                else
                    set(handles.parameter_value_box, 'Visible', 'off');
                    set(handles.parameter_name_list, 'Visible', 'off');
                    set(handles.use_parameter_button, 'Visible', 'off');                    
                end
                
            case 'write'
                exp_id = get(handles.experiment_box, 'value');
                if ~isempty(params{exp_id}),
                    if length(varargin) > 0,
                        save_param = merge_structs(params{exp_id}, propertylist2struct(varargin{:}));                        
                    else
                        save_param = params{exp_id};
                    end
                    assignin('base', 'gui_set_opts', save_param);                    
                else
                    if length(varargin) > 0,
                        assignin('base', 'gui_set_opts', propertylist2struct(varargin{:}));
                    end
                end                
        end
                
        
        
% --- Executes on button press in create_images_button.
function create_images_button_Callback(hObject, eventdata, handles)
% hObject    handle to create_images_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    if ask_for_correct_ival(handles),
        try
        set_button_state(handles, 'off', {''});
        evalin('base', ['p300gui_visualize_', get(handles.save_study, 'string')]);
            figure(handles.figure1);
            set_button_state(handles, 'on', {''});
        catch
            error = lasterror;
            disp(error.message);            
            set_button_state(handles, 'on', {''});
            add_to_message_box(handles, 'Visualization didn''t work. Is it implemented properly?');
        end
    else
        add_to_message_box(handles, 'Cancelled...');
    end
else
    add_to_message_box(handles, 'Please initialize and load data first...');
end

% --- Executes on button press in close_images_button.
function close_images_button_Callback(hObject, eventdata, handles)
% hObject    handle to close_images_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fh=findall(0,'Type','Figure');
for i = 1:length(fh),
    if ~strcmp(get(fh(i), 'name'), 'p300_gui'),
        close(fh(i));
    end
end

% --- Executes on button press in find_ivals_button.
function find_ivals_button_Callback(hObject, eventdata, handles)
% hObject    handle to find_ivals_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    evalin('base', 'opt.selectival = select_time_intervals(proc_selectIval(epo_r, [0 opt.ival(2)]), ''nIvals'',3, ''visualize'', 0, ''sort'', 1);');
    reset_ivals_from_base(handles);
else
    add_to_message_box(handles, 'Please initialize first and load data...');
end


function vp_code_box_Callback(hObject, eventdata, handles)
% hObject    handle to vp_code_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of vp_code_box as text
%        str2double(get(hObject,'String')) returns contents of vp_code_box as a double


% --- Executes during object creation, after setting all properties.
function vp_code_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to vp_code_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
global VP_CODE
if ~isempty(VP_CODE),
    set(hObject, 'string', VP_CODE);
end




% --- Executes on selection change in study_box.
function study_box_Callback(hObject, eventdata, handles)
% hObject    handle to study_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns study_box contents as cell array
%        contents{get(hObject,'Value')} returns selected item from study_box


% --- Executes during object creation, after setting all properties.
function study_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to study_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
p = mfilename('fullpath');
dlist = dir([fileparts(p) '/setups/']);
str = {};
matchStr = 'p300gui_setup_online_';
for i = 1:length(dlist),
    if ~dlist(i).isdir && ~isempty(strmatch(matchStr, dlist(i).name)),
        str{length(str)+1} = dlist(i).name(length(matchStr)+1:end-2);
    end
end
set(hObject, 'string', sort(str));


% --- Executes on button press in relist_files_button.
function relist_files_button_Callback(hObject, eventdata, handles)
% hObject    handle to relist_files_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
look_for_existing_files(handles);


% --- Executes on button press in individual_files_tick.
function individual_files_tick_Callback(hObject, eventdata, handles)
% hObject    handle to individual_files_tick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of individual_files_tick
look_for_existing_files(handles);

% --- Executes on button press in today_only_tick.
function today_only_tick_Callback(hObject, eventdata, handles)
% hObject    handle to today_only_tick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of today_only_tick
look_for_existing_files(handles)

% --- Executes on button press in reject_channels_tick.
function reject_channels_tick_Callback(hObject, eventdata, handles)
% hObject    handle to reject_channels_tick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of reject_channels_tick


% --- Executes on button press in create_features_button.
function create_features_button_Callback(hObject, eventdata, handles)
% hObject    handle to create_features_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    if ask_for_correct_ival(handles),
        drawnow;
        evalin('base', strcat('analyze=[];', ...
            'remainmessage='''';', ...
            'do_xval=0;', ...
            'p300gui_xval_', ...
            get(handles.save_study, 'string'), ...
            ';bbci.setup_opts = opt;', ...
            'analyze = struct(''features'', features, ''message'', remainmessage);', ...
            'bbci.analyze = analyze;'));      
    end
else
    add_to_message_box(handles, 'Please initialize first, load data and select intervals...');
end

% --- Executes on button press in xval_button.
function xval_button_Callback(hObject, eventdata, handles)
% hObject    handle to xval_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    if ask_for_correct_ival(handles),
        drawnow;
        classifiers = get(handles.choose_classifier_popup, 'string');
        sel_class = get(handles.choose_classifier_popup, 'value');
        if sel_class > 1,
            assignin('base', 'tmp', classifiers(sel_class));
            evalin('base', 'opt.model = tmp; clear tmp;');
            try
                add_to_message_box(handles, 'Starting x-validation. Please wait...');drawnow;
                set_button_state(handles, 'off', {''});
                evalin('base', strcat('analyze=[];', ...
                        'do_xval=1;', ...
                        'p300gui_xval_', ...
                        get(handles.save_study, 'string'), ...
                        ';bbci.setup_opts = opt;', ...
                        'analyze = struct(''features'', features, ''message'', remainmessage);', ...
                        'bbci.analyze = analyze;'));
                set_button_state(handles, 'on', {''});
                xval_result = evalin('base', 'xval_result');
                add_to_message_box(handles, sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f Crossvalidation done...',xval_result(1), xval_result(2)));
            catch
                error = lasterror;
                disp(error.message);                
                set_button_state(handles, 'on', {''});
                add_to_message_box(handles, 'Something went wrong and I didn''t do xvalidation. Please check the command window for errors');
            end        
        else
            add_to_message_box(handles, 'No classifier selected...');
        end    
    end
else
    add_to_message_box(handles, 'Please initialize first, load data and select intervals...');
end

function inited =  check_if_initialized(handles),
    inited = ~isempty(get(handles.save_study, 'string'));

function loaded = check_if_data_loaded(handles),
    loaded = evalin('base', 'exist(''Cnt'', ''var'')'); 
    
function val_file = find_classifier_name(handles),
    base_file = evalin('base', 'bbci.save_name');
    val_file = base_file;
    counter = 1;     
    while exist(strcat(val_file, '.mat'), 'file'),
        val_file = sprintf('%s_%i', base_file, counter);
        counter = counter+1;
    end
        

% --- Executes on button press in save_classifier_button.
function save_classifier_button_Callback(hObject, eventdata, handles)
% hObject    handle to save_classifier_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    if ask_for_correct_ival(handles),
        classifiers = get(handles.choose_classifier_popup, 'string');
        sel_class = get(handles.choose_classifier_popup, 'value');
        if sel_class > 1,
            old_name = evalin('base', 'bbci.save_name');
            assignin('base', 'tmp', find_classifier_name(handles));
            evalin('base', 'bbci.save_name = tmp; clear tmp;');
            assignin('base', 'tmp', classifiers(sel_class));
            evalin('base', 'opt.model = tmp; clear tmp;');
            try
                add_to_message_box(handles, 'Saving classifier. Please wait...');drawnow;
                set_button_state(handles, 'off', {''});
                evalin('base', strcat('analyze=[];', ...
                        'do_xval=0;', ...
                        'p300gui_xval_', ...
                        get(handles.save_study, 'string'), ...
                        ';bbci.setup_opts = opt;', ...
                        'bbci.start_marker = 251;', ...
                        'bbci.quit_marker = 254;', ...
                        'analyze = struct(''features'', features, ''message'', remainmessage);', ...
                        'bbci.analyze = analyze;', ...
                        'bbci_bet_finish;'));
                set_button_state(handles, 'on', {''});
                add_to_message_box(handles, 'Classifier written.');
            catch
                error = lasterror;
                disp(error.message);                
                set_button_state(handles, 'on', {''});
                add_to_message_box(handles, 'Something went wrong and I didn''t save the classifier. Please check the command window for errors');
            end    
            assignin('base', 'tmp', old_name);
            evalin('base', 'bbci.save_name = tmp; clear tmp;');
            look_for_existing_classifiers(handles);
        else
            add_to_message_box(handles, 'No classifier selected...');
        end            
    else
        add_to_messsage_box(handles, 'Cancelled...');
    end
else
    add_to_message_box(handles, 'Please initialize first, load data and select intervals...');
end

% --- Executes on selection change in choose_classifier_popup.
function choose_classifier_popup_Callback(hObject, eventdata, handles)
% hObject    handle to choose_classifier_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns choose_classifier_popup contents as cell array
%        contents{get(hObject,'Value')} returns selected item from choose_classifier_popup


% --- Executes during object creation, after setting all properties.
function choose_classifier_popup_CreateFcn(hObject, eventdata, handles)
% hObject    handle to choose_classifier_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on selection change in experiment_box.
function experiment_box_Callback(hObject, eventdata, handles)
% hObject    handle to experiment_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns experiment_box contents as cell array
%        contents{get(hObject,'Value')} returns selected item from experiment_box
handle_experiment_parameters(handles, 'update_exp');

% --- Executes during object creation, after setting all properties.
function experiment_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to experiment_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in simulate_tick.
function simulate_tick_Callback(hObject, eventdata, handles)
% hObject    handle to simulate_tick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of simulate_tick

% --- Executes on button press in clear_button.
function clear_button_Callback(hObject, eventdata, handles)
% hObject    handle to clear_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    evalin('base', 'clear Cnt epo epo_r opt mrk_orig mrk_rej mrk features analyze');
    cla(handles.interval_axes);
    colorbar off;
    reset_ivals(handles);
         
% --- Executes on button press in run_experiment_button.
function run_experiment_button_Callback(hObject, eventdata, handles)
% hObject    handle to run_experiment_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles),
    switch get(handles.run_experiment_switch, 'UserData'),
        case 'stop'
            try
                add_to_message_box(handles, 'Starting experiment...');
                set(handles.run_experiment_switch,'UserData','run');
                set(handles.run_experiment_button, 'string', 'Pause');
                experiments = get(handles.experiment_box, 'string');
                handle_experiment_parameters(handles, 'write', 'break_loop_handle', handles.run_experiment_switch);
                set_button_state(handles, 'off', {'stop_experiment_button', 'run_experiment_button'});
                evalin('base', strcat('do_action = ''', experiments{get(handles.experiment_box, 'value')}, ''';', ...
                    'simulate_run = ', num2str(get(handles.simulate_tick, 'value')), ';', ...
                    'p300gui_runscripts_', get(handles.save_study, 'string'), ';'));
                set_button_state(handles, 'on', {''});
                add_to_message_box(handles, 'Experiment done...');                
                if evalin('base', 'exist(''exp_res'');'),
                    exp_res = evalin('base', 'exp_res');
                    if isfield(exp_res, 'scores'),
                        add_to_message_box(handles, sprintf('Result: %0.2f%%', mean(exp_res.scores)*100));
                    end
                end
                set(handles.run_experiment_switch,'UserData','stop');     
                set_run_button_text(handles)
            catch
                error = lasterror;
                disp(error.message);
                set_button_state(handles, 'on', {''});
                set(handles.run_experiment_switch,'UserData','stop');
                set_run_button_text(handles)
                add_to_message_box(handles, 'I tried, but the experiment wouldn''t run. Please check the command window');
            end
        case 'pause'
            add_to_message_box(handles, 'Resuming experiment...');
            set(handles.run_experiment_switch,'UserData','run');
            set_run_button_text(handles);
        case 'run'
            add_to_message_box(handles, 'Pausing experiment...');
            set(handles.run_experiment_switch,'UserData','pause');
            set_run_button_text(handles);
    end
end

function set_run_button_text(handles),
    switch get(handles.run_experiment_switch, 'UserData'),
        case 'stop'
            set(handles.run_experiment_button, 'string', 'Run experiment');
        case 'pause'
            set(handles.run_experiment_button, 'string', 'Resume');
        case 'run'
            set(handles.run_experiment_button, 'string', 'Pause');
    end
    
function set_button_state(handles, state, exclude),
    flds = fieldnames(handles);
    for i = 1:length(flds),
        if strcmp(get(handles.(flds{i}), 'type'), 'uicontrol') && isempty(intersect(flds{i}, exclude)),
            set(handles.(flds{i}), 'enable', state);
        end
    end
    drawnow;
    

% --- Executes on button press in stop_experiment_button.
function stop_experiment_button_Callback(hObject, eventdata, handles)
% hObject    handle to stop_experiment_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set_button_state(handles, 'on', {''});
set(handles.run_experiment_switch,'UserData','stop');
set_run_button_text(handles);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Non callback functions (helper functions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function add_to_message_box(handles, str),
    curMes = get(handles.message_box, 'string');
    newMes = sprintf('%s\r%s', str, curMes);
    set(handles.message_box, 'string', newMes);

function look_for_existing_files(handles),
    global VP_CODE EEG_RAW_DIR TODAY_DIR
    if ~isempty(VP_CODE),
        set(handles.file_list_box, 'value', 1);
        val_file = evalin('base', 'bbci.allowed_files');
        files = {};
        if get(handles.today_only_tick, 'value'),
            if TODAY_DIR(end) == '\' || TODAY_DIR(end) == '/'
                [dum d(1).name] = fileparts(TODAY_DIR(1:end-1));
            else
                [dum d(1).name] = fileparts(TODAY_DIR);
            end
            val_id = 1;
        else
            d = dir(EEG_RAW_DIR);
            val_id = strmatch([VP_CODE '_'], {d(:).name});
        end
        % select valid dirs and sort
        d = d(val_id);
        [dum id] = sort({d(:).name});
        d = d(fliplr(id));         
        for d_id = 1:length(d),
            d2 = dir([EEG_RAW_DIR d(d_id).name filesep '*.eeg']);
            for f_id = 1:length(val_file),
                val_f_id = strmatch(val_file{f_id}, {d2(:).name});
                if ~isempty(val_f_id),
                    if get(handles.individual_files_tick, 'value')
                        val_f_id = flipud(val_f_id);
                        for j = 1:length(val_f_id),
                            files{length(files)+1} = [d(d_id).name filesep d2(val_f_id(j)).name(1:end-4)];
                        end
                    else
                        files{length(files)+1} = [d(d_id).name filesep d2(val_f_id(1)).name(1:end-4) '*'];
                    end
                end
            end
        end
        set(handles.file_list_box, 'string', files);
    end

function look_for_existing_classifiers(handles),
    global VP_CODE EEG_RAW_DIR TODAY_DIR
    set(handles.run_classifier_menu, 'value', 1);
    val_file = evalin('base', 'bbci.save_name');
    if val_file(end) == '\' || val_file(end) == '/', val_file = val_file(1:end-1), end;
    [dum val_file] = fileparts(val_file);
    val_file = {val_file};
    files = {};
    if get(handles.today_only_classifier_tick, 'value'),
        if TODAY_DIR(end) == '\' || TODAY_DIR(end) == '/'
            [dum d(1).name] = fileparts(TODAY_DIR(1:end-1));
        else
            [dum d(1).name] = fileparts(TODAY_DIR);
        end
        val_id = 1;
    else
        d = dir(EEG_RAW_DIR);
        val_id = strmatch([VP_CODE '_'], {d(:).name});
    end
    % select valid dirs and sort
    d = d(val_id);
    [dum id] = sort({d(:).name});
    d = d(fliplr(id));    
    for d_id = 1:length(d),
        d2 = dir([EEG_RAW_DIR d(d_id).name filesep '*.mat']);
        for f_id = 1:length(val_file),
            val_f_id = strmatch(val_file{f_id}, {d2(:).name});
            if ~isempty(val_f_id),
                % select valid files and sort
                d2 = d2(val_f_id);
                [dum id] = sort({d2(:).name});
                d2 = d2(fliplr(id));                 
                for j = 1:length(d2),
                    files{length(files)+1} = [d(d_id).name filesep d2(j).name(1:end-4)];
                end
            end
        end
    end
    if ~isempty(files),
        set(handles.run_classifier_menu, 'string', files);
        set(handles.reload_classifier_menu, 'string', files);
    else
        set(handles.run_classifier_menu, 'string', {'No matching classifier found.'});
        set(handles.reload_classifier_menu, 'string', {'No matching classifier found.'});
    end

function update_image_axis(handles),
    axes(handles.interval_axes);
    epo_r = evalin('base', 'epo_r');
    visualize_score_matrix(epo_r, '', struct(), handles);

function visualize_score_matrix(epo_r, nfo, opt, handles)
    [opt_visu, isdefault]= ...
        set_defaults(opt, ...
                     'clf', 0, ...
                     'colormap', cmap_posneg(51), ...
                     'mark_clab', {'Fz','FCz','Cz','CPz','Pz','Oz'}, ...
                     'xunit', 'ms');
    cla(handles.interval_axes);
    % order channels for visualization:
    %  scalp channels first, ordered from frontal to occipital (as returned
    %  by function scalpChannels),
    %  then non-scalp channels
    clab= clab_in_preserved_order(scalpChannels, strhead(epo_r.clab));
    clab_nonscalp= clab_in_preserved_order(epo_r, ...
                           setdiff(strhead(epo_r.clab), scalpChannels));
    epo_r= proc_selectChannels(epo_r, cat(2, clab, clab_nonscalp)); 
    axes(handles.interval_axes);
    colormap(opt_visu.colormap);
    start_id = find(epo_r.t == 0);
    imagesc(epo_r.t(start_id:end), 1:length(epo_r.clab), epo_r.x(start_id:end,:)'); 
    set(gca, 'CLim',[-1 1]*max(abs(epo_r.x(:)))); 
    colorbar;
    cidx= strpatternmatch(opt_visu.mark_clab, epo_r.clab);
    set(gca, 'YTick',cidx, 'YTickLabel',opt_visu.mark_clab, ...
              'TickLength',[0.005 0]);
    if isdefault.xunit & isfield(epo_r, 'xUnit'),
      opt_visu.xunit= epo_r.xUnit;
    end
    xlabel(['[' opt_visu.xunit ']']);
    ylabel('channels');
    set(gca, 'YLim', [-2 length(epo_r.clab)+2]);
    ylimits= get(gca, 'YLim');
    ylimits = ylimits+[1 -1];
%     set(gca, 'YLim',ylimits+[-2 2], 'NextPlot','add');
    for ii= 1:5,
        xx= [-10 -10] + [-1 1]*1000/epo_r.fs/2;
        H.box(:,ii)= line(xx([1 2; 2 2; 2 1; 1 1; 1 2]), ...
                      ylimits([1 1; 1 2; 2 2; 2 1; 1 1]), ...
                      'color',[0 0.5 0], 'LineWidth',0.5);
    end
    assignin('base', 'H', H);
    
function reset_ivals_from_base(handles),
    set_proper_range_ival(handles);
    ivals = evalin('base', 'opt.selectival');
    set_ivals(handles, ivals);
    
function reset_ivals(handles),
    box_name = {'start', 'end'};
    for ival_nr = 1:5,
        for str_end = 1:2,
            set(handles.(['ival_' box_name{str_end} '_' num2str(ival_nr)]), 'value', 1);
        end        
    end
    
function set_ivals(handles, ivals),
    reset_ivals(handles);
    box_name = {'start', 'end'};
    for ival_nr = 1:size(ivals, 1),
        if ival_nr <= size(ivals, 1),
            for str_end = 1:2,
               ran_val = get(handles.(['ival_' box_name{str_end} '_' num2str(ival_nr)]), 'string');
                sel_id = strmatch(num2str(ivals(ival_nr, str_end)), ran_val, 'exact');
                set(handles.(['ival_' box_name{str_end} '_' num2str(ival_nr)]), 'value', sel_id);
            end
        else
            set(handles.(['ival_' box_name{str_end} '_' num2str(ival_nr)]), 'value', 1);
        end
    end
    redraw_intervals(handles);
    
function set_proper_range_ival(handles),
    range = evalin('base', 'epo_r.t');
    range = range(range >= 0);
    range = {'NaN', range(:)};
    
    for i = 1:5,
        set(handles.(['ival_start_' num2str(i)]), 'string', range);
        set(handles.(['ival_end_' num2str(i)]), 'string', range);
    end
    
function redraw_intervals(handles),
    H = evalin('base', 'H');
    fs = evalin('base', 'epo_r.fs');
    shift_len = 1000/fs/2;
    ivals = get_ivals(handles, 0);
    range = get(handles.ival_start_1, 'string');
    for i = 1:size(ivals, 1),
        if any(isnan(ivals(i,:))),
            set(H.box(i), 'XData', ones(1,5)*-10);
        else
            set(H.box(i), 'YData', get(H.box(1), 'YData'));
            set(H.box(i), 'XData', [ivals(i,1)-shift_len ivals(i,2)+shift_len ivals(i,2)+shift_len ivals(i,1)-shift_len ivals(i,1)-shift_len]);
        end
    end
    
function ivals = get_ivals(handles, pruneNan),
    range = get(handles.ival_start_1, 'string');
    ivals = ones(5,2)*NaN;
    for i = 1:5,
        ivals(i,:) = [str2num(str2mat(range(get(handles.(['ival_start_' num2str(i)]), 'value')))) str2num(str2mat(range(get(handles.(['ival_end_' num2str(i)]), 'value'))))];
    end   
    if pruneNan,
        ivals(any(isnan(ivals),2),:) = [];
    end
     
function transfer_ivals_to_base(handles),
    ivals = get_ivals(handles, 1);
    assignin('base', 'tmp', sort(ivals,1)); 
    evalin('base', 'opt.selectival = tmp; clear tmp');
        
function success = ask_for_correct_ival(handles),
    local_ival = sort(get_ivals(handles, 1));
    base_ival = sort(evalin('base', 'opt.selectival'));
    if ~isequal(local_ival, base_ival),
        button = questdlg(sprintf('You changed the intervals, but didn''t save.\nDo you want to use the new intervals?'), 'Warning');
        switch button
            case 'Yes'
                transfer_ivals_to_base(handles);
                success = 1;
            case 'No'
                reset_ivals_from_base(handles)
                success = 1;
            case 'Cancel'
                success = 0;
        end
    else
        success = 1;
    end
 
function reset_channel_names(handles),
    if check_if_initialized(handles) && check_if_data_loaded(handles),
        clab_loaded = evalin('base', 'epo.clab');
        set(handles.channel_names_listbox, 'value', 1);
        set(handles.channel_names_listbox, 'string', clab_loaded);
    end

function update_nr_chan_selected(handles),
    str = sprintf('%i of %i channels selected', length(get(handles.channel_names_listbox, 'value')), length(get(handles.channel_names_listbox, 'string')));
    set(handles.selected_channels_text, 'string', str);    

    
function reset_axes(handles);
    ax = handles.interval_axes;
    cla(ax, 'reset');
    set(ax, 'XTick', []);
    set(ax, 'YTick', []);
    
        
% --- Executes on selection change in popupmenu14.
function popupmenu14_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu14 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu14


% --- Executes during object creation, after setting all properties.
function popupmenu14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on selection change in run_classifier_menu.
function run_classifier_menu_Callback(hObject, eventdata, handles)
% hObject    handle to run_classifier_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns run_classifier_menu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from run_classifier_menu


% --- Executes during object creation, after setting all properties.
function run_classifier_menu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to run_classifier_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in start_classifier_button.
function start_classifier_button_Callback(hObject, eventdata, handles)
% hObject    handle to start_classifier_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global EEG_RAW_DIR VP_CODE TODAY_DIR
if check_if_initialized(handles),
    classifiers = get(handles.run_classifier_menu, 'string');
    sel_class = get(handles.run_classifier_menu, 'value');
    if ~strcmp(classifiers{sel_class}, 'Initialize first') && ~isempty(classifiers{sel_class}) && ~strcmp(classifiers{sel_class}, 'No matching classifier found.'),  
%         settings_bbci= {'quit_marker', 254};
        cmd_init= sprintf('VP_CODE= ''%s''; TODAY_DIR= ''%s'';set_general_port_fields(''localhost'');', VP_CODE, TODAY_DIR);
        bbci_cfy= [EEG_RAW_DIR classifiers{sel_class} '.mat'];
        cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''')'];
        system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']); 
    end
end

% --- Executes on button press in stop_classifier_button.
function stop_classifier_button_Callback(hObject, eventdata, handles)
% hObject    handle to stop_classifier_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ppTrigger(254);



% --- Executes during object creation, after setting all properties.
function create_features_button_CreateFcn(hObject, eventdata, handles)
% hObject    handle to create_features_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called




% --- Executes on button press in today_only_classifier_tick.
function today_only_classifier_tick_Callback(hObject, eventdata, handles)
% hObject    handle to today_only_classifier_tick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of today_only_classifier_tick
look_for_existing_classifiers(handles);



% --- Executes on selection change in channel_selection_list_box.
function channel_selection_list_box_Callback(hObject, eventdata, handles)
% hObject    handle to channel_selection_list_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns channel_selection_list_box contents as cell array
%        contents{get(hObject,'Value')} returns selected item from channel_selection_list_box


% --- Executes during object creation, after setting all properties.
function channel_selection_list_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to channel_selection_list_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function number_channel_box_Callback(hObject, eventdata, handles)
% hObject    handle to number_channel_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of number_channel_box as text
%        str2double(get(hObject,'String')) returns contents of number_channel_box as a double


% --- Executes during object creation, after setting all properties.
function number_channel_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to number_channel_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
str = {'Number of channels'};
for i = 1:64,
    str{i+1} = i;
end
set(hObject, 'string', str, 'value', 17);


% --- Executes on button press in run_selection_button.
function run_selection_button_Callback(hObject, eventdata, handles)
% hObject    handle to run_selection_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    if ask_for_correct_ival(handles),
        method = get(handles.channel_selection_list_box, 'string');
        method = method{get(handles.channel_selection_list_box, 'value')};
        nrCh = get(handles.number_channel_box, 'string');
        nrCh = nrCh{get(handles.number_channel_box, 'value')};
        
        set_button_state(handles, 'off', {''});
        try
            add_to_message_box(handles, 'Starting channel selection...');drawnow;
            evalin('base', strcat('analyze=[];', ...
                'remainmessage='''';', ...
                'do_xval=0;', ...
                'p300gui_xval_', ...
                get(handles.save_study, 'string'), ...
                ';bbci.setup_opts = opt;', ...
                'analyze = struct(''features'', features, ''message'', remainmessage);', ...
                'bbci.analyze = analyze;')); 

            switch method,
                case 'SWLDA - featurewise',
                    selected = evalin('base', strcat('proc_stepwiseChannelSelect(', ...
                        'features.x, features.y, ', ...
                        '''featPChan'', ', num2str(size(get_ivals(handles, 1), 1)), ',', ...
                        '''maxChan'', ', num2str(nrCh), ',', ...
                        '''visualize'', 0,', ...
                        '''channelwise'', 0);'));

                case 'SWLDA - channelwise',
                    selected = evalin('base', strcat('proc_stepwiseChannelSelect(', ...
                        'features.x, features.y, ', ...
                        '''featPChan'', ', num2str(size(get_ivals(handles, 1), 1)), ',', ...
                        '''maxChan'', ', num2str(nrCh), ',', ...
                        '''visualize'', 0,', ...
                        '''channelwise'', 1);'));
            end
            reset_channel_names(handles);
            set(handles.channel_names_listbox, 'value', selected);
            update_nr_chan_selected(handles);
            add_to_message_box(handles, ['Selected ' num2str(length(selected)) ' channels.']);
            set_button_state(handles, 'on', {''});
        catch
            error = lasterror;
            disp(error.message);            
            set_button_state(handles, 'on', {''});
            add_to_message_box(handles, 'Something went wrong. I''m sorry!');
        end
    end
    
else
    add_to_message_box(handles, 'Please initialize and load data first.');
end


% --- Executes on button press in show_channels_button.
function show_channels_button_Callback(hObject, eventdata, handles)
% hObject    handle to show_channels_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    clab = get(handles.channel_names_listbox, 'string')';
    sel_clab = get(handles.channel_names_listbox, 'value')';
    mnt = getElectrodePositions(clab);
    colOrder = [0.9 0 0.9; 0.4 0.57 1];
    labelProps = {'FontName','Times','FontSize',8,'FontWeight','normal'};
    markerProps = {'MarkerSize', 15, 'MarkerEdgeColor','k','MarkerFaceColor',[1 1 1]};
    highlightProps = {'MarkerEdgeColor','k','MarkerFaceColor',colOrder(1,:),...
        'LineWidth',2};
    linespec = {'Color' 'k' 'LineWidth' 2};
    refProps = {'FontSize', 8, 'FontName', 'Times','BackgroundColor',[.8 .8 .8],'HorizontalAlignment','center','Margin',2};

    opt = {'showLabels',1,'labelProps',labelProps,'markerProps',...
        markerProps,'markChans',clab(sel_clab),'markMarkerProps',...
        highlightProps,'linespec',linespec,'ears',1,'reference','nose', ...
        'referenceProps', refProps};

    % Draw the stuff
    fig = figure;
    H= drawScalpOutline(mnt, opt{:});
    set(fig, 'MenuBar', 'none');
    set(gca,'box','on')
    set(gca, 'Position', [0 0 1 1]);
    pos = get(gcf,'Position');
    axis off;
end

% --- Executes on button press in use_channels_button.
function use_channels_button_Callback(hObject, eventdata, handles)
% hObject    handle to use_channels_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if check_if_initialized(handles) && check_if_data_loaded(handles),
    sel_clab = get(handles.channel_names_listbox, 'value')';
    assignin('base', 'tmp', sel_clab);
    evalin('base', 'epo = proc_selectChannels(epo, tmp);');
    evalin('base', 'epo_r = proc_selectChannels(epo_r, tmp);');
    evalin('base', 'clear tmp;');
    reset_channel_names(handles);
    set(handles.channel_names_listbox, 'value', [1:length(sel_clab)]);
    update_nr_chan_selected(handles);    
    update_image_axis(handles);
    reset_ivals_from_base(handles);
end



% --- Executes on selection change in channel_names_listbox.
function channel_names_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to channel_names_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns channel_names_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from channel_names_listbox
update_nr_chan_selected(handles);

% --- Executes during object creation, after setting all properties.
function channel_names_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to channel_names_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes during object creation, after setting all properties.
function interval_axes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to interval_axes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate interval_axes
set(hObject, 'XTick', []);
set(hObject, 'YTick', []);
text(1,  .97, 'Initialize --> ', 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10);
text(1,  .7, 'Load data --> ', 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right', 'FontSize', 10);


% --- Executes on selection change in parameter_name_list.
function parameter_name_list_Callback(hObject, eventdata, handles)
% hObject    handle to parameter_name_list (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns parameter_name_list contents as cell array
%        contents{get(hObject,'Value')} returns selected item from parameter_name_list
handle_experiment_parameters(handles, 'update');

% --- Executes during object creation, after setting all properties.
function parameter_name_list_CreateFcn(hObject, eventdata, handles)
% hObject    handle to parameter_name_list (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function parameter_value_box_Callback(hObject, eventdata, handles)
% hObject    handle to parameter_value_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of parameter_value_box as text
%        str2double(get(hObject,'String')) returns contents of parameter_value_box as a double


% --- Executes during object creation, after setting all properties.
function parameter_value_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to parameter_value_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in use_parameter_button.
function use_parameter_button_Callback(hObject, eventdata, handles)
% hObject    handle to use_parameter_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handle_experiment_parameters(handles, 'save');




% --- Executes on button press in pushbutton27.
function pushbutton27_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handle_experiment_parameters(handles, 'write', 'taradsfa', 23);




% --- Executes on selection change in reload_classifier_menu.
function reload_classifier_menu_Callback(hObject, eventdata, handles)
% hObject    handle to reload_classifier_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns reload_classifier_menu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from reload_classifier_menu


% --- Executes during object creation, after setting all properties.
function reload_classifier_menu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to reload_classifier_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in reload_ivals_button.
function reload_ivals_button_Callback(hObject, eventdata, handles)
% hObject    handle to reload_ivals_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global EEG_RAW_DIR
class_name = get(handles.reload_classifier_menu, 'string');
class_id = get(handles.reload_classifier_menu, 'value');
file = [EEG_RAW_DIR class_name{class_id} '.mat'];
bbci_load = load(file, 'bbci');restore_ival = bbci_load.bbci.setup_opts.selectival;
set_ivals(handles, restore_ival);




