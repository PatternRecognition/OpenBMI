function varargout = paradigm_MI(varargin)
% PARADIGM_MI MATLAB code for paradigm_MI.fig
%      PARADIGM_MI, by itself, creates a new PARADIGM_MI or raises the existing
%      singleton*.
%
%      H = PARADIGM_MI returns the handle to a new PARADIGM_MI or the handle to
%      the existing singleton*.
%
%      PARADIGM_MI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PARADIGM_MI.M with the given input arguments.
%
%      PARADIGM_MI('Property','Value',...) creates a new PARADIGM_MI or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before paradigm_MI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to paradigm_MI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help paradigm_MI

% Last Modified by GUIDE v2.5 21-Nov-2016 14:28:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @paradigm_MI_OpeningFcn, ...
    'gui_OutputFcn',  @paradigm_MI_OutputFcn, ...
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

% --- Executes just before paradigm_MI is made visible.
function paradigm_MI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to paradigm_MI (see VARARGIN)
set(gcf,'units','points','position',[350 200 490 470])
% Choose default command line output for paradigm_MI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

initialize_gui(hObject, handles, false);

% UIWAIT makes paradigm_MI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = paradigm_MI_OutputFcn(hObject, eventdata, handles)
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

% handles.metricdata.density = 0;
% handles.metricdata.volume  = 0;
%
% set(handles.density, 'String', handles.metricdata.density);
% set(handles.volume,  'String', handles.metricdata.volume);
% set(handles.mass, 'String', 0);
%
% set(handles.unitgroup, 'SelectedObject', handles.english);
%
% set(handles.text4, 'String', 'lb/cu.in');
% set(handles.text5, 'String', 'cu.in');
% set(handles.text6, 'String', 'lb');

% Update handles structure
guidata(handles.figure1, handles);
set(handles.figure1, 'Name', 'Motor Imagery Paradigm');


% --- Executes during object creation, after setting all properties.
function paradigm_CreateFcn(hObject, eventdata, handles)
% hObject    handle to paradigm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% axes(hObject)
% g = imread('paradigm.png','BackgroundColor',[0.941 0.941 0.941]);
% image(g)
% axis off
% axis image
% Hint: place code in OpeningFcn to populate paradigm

%% Make a paradigm
%%
function TimeStimulus_Callback(hObject, eventdata, handles)
% hObject    handle to TimeStimulus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeStimulus as text
%        str2double(get(hObject,'String')) returns contents of TimeStimulus as a double

% --- Executes during object creation, after setting all properties.
function TimeStimulus_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TimeStimulus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%
function TimeIsi_Callback(hObject, eventdata, handles)
% hObject    handle to TimeIsi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeIsi as text
%        str2double(get(hObject,'String')) returns contents of TimeIsi as a double

% --- Executes during object creation, after setting all properties.
function TimeIsi_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TimeIsi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%
function TimeRest_Callback(hObject, eventdata, handles)
% hObject    handle to TimeRest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeRest as text
%        str2double(get(hObject,'String')) returns contents of TimeRest as a double

% --- Executes during object creation, after setting all properties.
function TimeRest_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TimeRest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function NumOfTrials_Callback(hObject, eventdata, handles)
% hObject    handle to NumOfTrials (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NumOfTrials as text
%        str2double(get(hObject,'String')) returns contents of NumOfTrials as a double

% --- Executes during object creation, after setting all properties.
function NumOfTrials_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NumOfTrials (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function NumOfClass_Callback(hObject, eventdata, handles)
% hObject    handle to NumOfClass (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NumOfClass as text
%        str2double(get(hObject,'String')) returns contents of NumOfClass as a double

% --- Executes during object creation, after setting all properties.
function NumOfClass_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NumOfClass (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%% Start the paradigm
% --- Executes on button press in MakeParadigm.
function MakeParadigm_Callback(hObject, eventdata, handles)
% hObject    handle to MakeParadigm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t_stimulus t_isi t_rest N_trial N_class

t_stimulus = str2double(get(handles.TimeStimulus , 'String'));
t_isi=  str2double(get(handles.TimeIsi , 'String'));
t_rest=  str2double(get(handles.TimeRest , 'String'));
N_trial=  str2double(get(handles.NumOfTrials , 'String'));
N_class=  str2double(get(handles.NumOfClass , 'String'));

Makeparadigm_MI({'time_sti',t_stimulus,'time_isi',t_isi,'time_rest',t_rest,'num_trial',N_trial,'num_class',N_class,...
    'time_jitter',0.1,'num_screen',2});



%% Offline Analysis
%% Load data
% --- Executes on button press in AddData.
function AddData_Callback(hObject, eventdata, handles)
% hObject    handle to AddData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global EEG
[file,path]=uigetfile('*.eeg','Load EEG file (.eeg)');
[file]=strsplit(file,'.eeg');
[EEG.data, EEG.marker, EEG.info]=Load_EEG([path,file{1}],{'device','brainVision'});

prompt={'Enter trigger number and corresponding class name:','Enter sampling frequency:'};
defaultans = {'{''1'',''right'';''2'',''left'';''3'',''foot''}','100'};
c=inputdlg(prompt,'Marker',[1 70],defaultans);
cls=c{1};
fs=str2double(c{2});


try
    temp=eval(cls);temp(:,3)=temp(:,1);temp(:,1)=[];
    opt_cellToStruct(temp);
catch
    warndlg(sprintf('Trigger and class information should be in a right form')),return
end
if any(~ismember(str2double(temp(:,2)),unique(EEG.marker.y)))
    warndlg(sprintf('Check your trigger number')),return
end
temp=EEG.info.fs/fs;
if temp~=round(temp) || temp<1
    warndlg(sprintf('(Original fs)/(input fs) should be an integer')),return
end


[EEG.data, EEG.marker, EEG.info]=Load_EEG([path,file{1}],{'device','brainVision';'marker',eval(cls);'fs',fs});
handles.EEG = EEG;
guidata(hObject, handles)
msgbox('Data load!');


function Bandpass1_Callback(hObject, eventdata, handles)
% hObject    handle to Bandpass1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Bandpass1 as text
%        str2double(get(hObject,'String')) returns contents of Bandpass1 as a double

% --- Executes during object creation, after setting all properties.
function Bandpass1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Bandpass1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Bandpass2_Callback(hObject, eventdata, handles)
% hObject    handle to Bandpass2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Bandpass2 as text
%        str2double(get(hObject,'String')) returns contents of Bandpass2 as a double


% --- Executes during object creation, after setting all properties.
function Bandpass2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Bandpass2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in SpatialFilter.
function SpatialFilter_Callback(hObject, eventdata, handles)
% hObject    handle to SpatialFilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns SpatialFilter contents as cell array
%        contents{get(hObject,'Value')} returns selected item from SpatialFilter


% --- Executes during object creation, after setting all properties.
function SpatialFilter_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SpatialFilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function TimeInterval1_Callback(hObject, eventdata, handles)
% hObject    handle to TimeInterval1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeInterval1 as text
%        str2double(get(hObject,'String')) returns contents of TimeInterval1 as a double


% --- Executes during object creation, after setting all properties.
function TimeInterval1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TimeInterval1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function TimeInterval2_Callback(hObject, eventdata, handles)
% hObject    handle to TimeInterval2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeInterval2 as text
%        str2double(get(hObject,'String')) returns contents of TimeInterval2 as a double


% --- Executes during object creation, after setting all properties.
function TimeInterval2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TimeInterval2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in selectFE.
function selectFE_Callback(hObject, eventdata, handles)
% hObject    handle to selectFE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectFE contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectFE


% --- Executes during object creation, after setting all properties.
function selectFE_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectFE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in selectCLF.
function selectCLF_Callback(hObject, eventdata, handles)
% hObject    handle to selectCLF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectCLF contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectCLF


% --- Executes during object creation, after setting all properties.
function selectCLF_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectCLF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in RemoveData.
function RemoveData_Callback(hObject, eventdata, handles)
% hObject    handle to RemoveData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Clear Offline analysis
% --- Executes on button press in ClearOfflineAnalysis.
function ClearOfflineAnalysis_Callback(hObject, eventdata, handles)
% hObject    handle to ClearOfflineAnalysis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global CNT SMT SelectFE SelectClassifier CSP
DataEEG = handles.EEG;
% Setting the data
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({DataEEG.data, DataEEG.marker, DataEEG.info}, field);
%--------------------------------------------------------------------------
% Load the selected frequency band
frequencyBand_1 = str2double(get(handles.Bandpass1,'String'));
frequencyBand_2 = str2double(get(handles.Bandpass2,'String'));
frequencyBand = [frequencyBand_1 frequencyBand_2];
%--------------------------------------------------------------------------
% Load the selected time interval
TimeInterval_1 = str2double(get(handles.TimeInterval1,'String'));
TimeInterval_2 = str2double(get(handles.TimeInterval2,'String'));
TimeInterval = [TimeInterval_1 TimeInterval_2];
% Process the band-pass filter
CNT = prep_filter(CNT,{'frequency' , frequencyBand});

% %--------------------------------------------------------------------------
% Process the spatial filter and segment the EEG data
spatialFilter = cellstr(get(handles.SpatialFilter,'String'));
SelectSpatialFilter = spatialFilter{get(handles.SpatialFilter,'Value')};
% Laplacian Filter
if strcmp(SelectSpatialFilter,'Laplacian')
    prompt={'Enter channel names to apply Laplacian filter:','Enter filter type:'};
    defaultans = {'{''C3'',''C4''}','small'};
    c=inputdlg(prompt,'Laplacian filter',[1 70],defaultans);
    option={'Channel',eval(c{1});'filterType',c{2}};
    CNT_LAP = prep_laplacian(filtCNT , option);
    SMT = prep_segmentation(CNT_LAP , {'interval' , TimeInterval});
    % CAR filter
elseif strcmp(SelectSpatialFilter, 'Common Average Reference')
    prompt={'Enter channel names to apply Common Average Reference filter:','Enter filter type:'};
    defaultans = {'{''Cz''}','incChan'};
    c=inputdlg(prompt,'Common Average Reference filter',[1 70],defaultans);
    option={'Channel',eval(c{1});'filterType',c{2}};
    [CNT_CAR] = prep_commonAverageReference(filtCNT , option);
    SMT = prep_segmentation(CNT_CAR , {'interval' , TimeInterval});
    
    % % Common Spatial pattern
    % elseif strcmp(SelectSpatialFilter, 'Common Spatial Pattern')
    %     SMT = prep_segmentation(filtCNT , {'interval' , TimeInterval});
    %     [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
else % non
    SMT = prep_segmentation(CNT , {'interval' , TimeInterval});
end


DataSMT = SMT;

FE = cellstr(get(handles.selectFE,'String'));
SelectFE = FE{get(handles.selectFE,'Value')};

switch SelectFE
    case 'Log-variance'
        FeatureSMT=func_featureExtraction(DataSMT, {'feature','logvar'});
    case 'CSP'
        [DataSMT, CSP_W, CSP_D]=func_csp(DataSMT,{'nPatterns', [3]});
        % 일단 2-class에 맞춰서
        cls_name=DataSMT.class(:,2);
        CSP{1,1}=CSP_W; CSP{1,2}=sprintf('%s vs %s',cls_name{1},cls_name{2});
        
        FeatureSMT=func_featureExtraction(DataSMT, {'feature','logvar'});
end


Classifier = cellstr(get(handles.selectCLF,'String'));
SelectClassifier = Classifier{get(handles.selectCLF,'Value')};
if strcmp (SelectClassifier,'Linear Discriminant Analysis (LDA)')
    clf_param=func_train(FeatureSMT,{'classifier','LDA'});
    
    %     % 2-class
    LDA = {};
    LDA{1,1}=clf_param;
    cls_name=DataSMT.class(:,2);
    LDA{1,2}=sprintf('%s vs %s',cls_name{1},cls_name{2});
    
elseif strcmp (SelectClassifier,'Suppot Vector Machine (SVM)')
    
end

% % Evaluation. Load the type of evaluation
% Evaluation = cellstr(get(handles.selectEvaluation,'String'));
% SelectEvaluation = Evaluation{get(handles.selectEvaluation,'Value')};
handles.fs = CNT.fs;
handles.frequencyBand = frequencyBand;
handles.Classifier = LDA;
handles.CSP = CSP;
guidata(hObject, handles)
msgbox('Complete the classification');



%% Real-time classification output
% --- Executes on button press in ConnetDevice.
function ConnetDevice_Callback(hObject, eventdata, handles)
% hObject    handle to ConnetDevice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global CSP LDA frequencyBand fs
frequencyBand = handles.frequencyBand;
LDA = handles.Classifier;
CSP = handles.CSP;
fs = handles.fs;

bbci_acquire_bv('close')
params = struct;
state = bbci_acquire_bv('init', params);

buffer_size=5000;
data_size=1500;
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));

% escapeKey = KbName('esc');
% waitKey=KbName('s');

a=1; cf_out=[];
tic
while true
%     [ keyIsDown, seconds, keyCode ] = KbCheck;
%     if keyIsDown
%         if keyCode(escapeKey)
%             
%         elseif keyCode(waitKey)
%             
%         end
%     end
    
    data = bbci_acquire_bv(state);
    orig_Dat = [orig_Dat;data];
    if length(orig_Dat)>buffer_size % prevent overflow
        Dat=orig_Dat(end-buffer_size+1:end,:);
        Dat2.x=Dat;
        Dat2.fs=state.fs;
        Dat=prep_resample(Dat2,500);
        Dat=Dat.x;
        %     if toc>feedback_t
        a=a+1;
        fDat=prep_filter(Dat, {'frequency', frequencyBand;'fs',fs});
        
        %=======================
        % add channel selection
        %=======================
        fDat=fDat(end-data_size:end,:);
        tm=func_projection(fDat, CSP);
        ft=func_featureExtraction(tm, {'feature','logvar'});
        [cf_out]=func_predict(ft, LDA)
        
    end
    set(handles.classficationOutput, 'String', cf_out);
    pause(0.005)
end


% --- Executes during object creation, after setting all properties.
function classficationOutput_CreateFcn(hObject, eventdata, handles)
% hObject    handle to classficationOutput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function classType_Callback(hObject, eventdata, handles)
% hObject    handle to classType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of classType as text
%        str2double(get(hObject,'String')) returns contents of classType as a double


% --- Executes during object creation, after setting all properties.
function classType_CreateFcn(hObject, eventdata, handles)
% hObject    handle to classType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

