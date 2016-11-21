function varargout = OpenBMI_GUI(varargin)
% OPENBMI_GUI MATLAB code for OpenBMI_GUI.fig
%      OPENBMI_GUI, by itself, creates a new OPENBMI_GUI or raises the existing
%      singleton*.
%
%      H = OPENBMI_GUI returns the handle to a new OPENBMI_GUI or the handle to
%      the existing singleton*.
%
%      OPENBMI_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in OPENBMI_GUI.M with the given input arguments.
%
%      OPENBMI_GUI('Property','Value',...) creates a new OPENBMI_GUI or raises
%      the existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before OpenBMI_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to OpenBMI_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help OpenBMI_GUI

% Last Modified by GUIDE v2.5 21-Nov-2016 14:30:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @OpenBMI_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @OpenBMI_GUI_OutputFcn, ...
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

% --- Executes just before OpenBMI_GUI is made visible.
function OpenBMI_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to OpenBMI_GUI (see VARARGIN)
set(gcf,'units','points','position',[350 200 580 560])
global EEG
% Choose default command line output for OpenBMI_GUI
handles.output = hObject;

% Update handles structure

guidata(hObject, handles);

initialize_gui(hObject, handles, false);

% UIWAIT makes OpenBMI_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = OpenBMI_GUI_OutputFcn(hObject, eventdata, handles)
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

set(handles.prepClear,'Enable','off');
set(handles.FEClear,'Enable','off');
set(handles.CLSClear,'Enable','off');
set(handles.Preprocessing,'visible','off');
set(handles.FeatureExtraction,'visible','off');
set(handles.Classifier,'visible','off');
set(handles.Results,'visible','off');
set(handles.visualization,'visible','off');
set(handles.Introduction,'visible','on');
% Update handles structure
guidata(handles.figure1, handles);
set(handles.figure1, 'Name', 'OpenBMI');
%% Show the LOGO
% --- Executes during object creation, after setting all properties.
function LOGO_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LOGO (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

axes(hObject)
g = imread('logo.png','BackgroundColor',[0.941 0.941 0.941]);
image(g)
axis off
axis image

% Hint: place code in OpeningFcn to populate LOGO



%% title "OpenBMI"
function title_Callback(hObject, eventdata, handles)
% hObject    handle to title (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of title as text
%        str2double(get(hObject,'String')) returns contents of title as a double

% --- Executes during object creation, after setting all properties.
function title_CreateFcn(hObject, eventdata, handles)
% hObject    handle to title (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



%% Title "Github"
function github_title_Callback(hObject, eventdata, handles)
% hObject    handle to github_title (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of github_title as text
%        str2double(get(hObject,'String')) returns contents of github_title as a double

% --- Executes during object creation, after setting all properties.
function github_title_CreateFcn(hObject, eventdata, handles)
% hObject    handle to github_title (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



%% Title Load data file
% --- Executes during object creation, after setting all properties.
function DataFile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DataFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

%% Title Data analysis
% --- Executes during object creation, after setting all properties.
function DataAnalysis_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DataAnalysis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% function DataFile_Callback(hObject, eventdata, handles)


%% Data information
% --- Executes during object creation, after setting all properties.
function DATALIST_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DATALIST (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

function DATALIST_Callback(hObject, eventdata, handles)
% hObject    handle to DATALIST (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



%% Push button ADD (Load file)
% --- Executes on button press in ADD.
function ADD_Callback(hObject, eventdata, handles)
% hObject    handle to ADD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global EEG
[file,path]=uigetfile('*.eeg','Load EEG file (.eeg)');
[file]=strsplit(file,'.eeg');
[EEG.data, EEG.marker, EEG.info]=Load_EEG([path,file{1}],{'device','brainVision'});
% data information
fileName=file{1};
dataSize=size(EEG.data.x);
fs=EEG.info.fs;
triggerType=unique(EEG.marker.y);
ss=sprintf('%s',fileName);
set(handles.DataName, 'String', ss);

str=sprintf('Data size: %.0fx%.0f\nNumber of channels: %d\n',dataSize(1),dataSize(2),dataSize(2));
str2=sprintf('Frequency: %d Hz\nTrigger type: \n%s',fs,mat2str(triggerType));
% handles.DATALIST.String=cat(2,str,str2);
set(handles.DATALIST, 'String', cat(2,str,str2));

prompt={'Enter trigger number and corresponding class name:','Enter sampling frequency:'};
defaultans = {'{''1'',''right'';''2'',''left''}','100'};
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
marker=str2double(EEG.marker.class(:,1)');
className=EEG.marker.class(:,2)';
numberTrial=size(EEG.marker.y_dec,2);
str2=sprintf('Frequency: %d Hz\nTrigger number: %s\nClass: %s\nNumber of trials: %d',fs,num2str(marker),strjoin(className),numberTrial);
set(handles.DATALIST, 'String', cat(2,str,str2));
handles.EEG = EEG;
guidata(hObject, handles)

%% Show the Data Name
% --- Executes on selection change in DataName.
function DataName_Callback(hObject, eventdata, handles)
% hObject    handle to DataName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns DataName contents as cell array
%        contents{get(hObject,'Value')} returns selected item from DataName


% --- Executes during object creation, after setting all properties.
function DataName_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DataName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%% Push button REMOVE (Load file)
% --- Executes on button press in REMOVE.
function REMOVE_Callback(hObject, eventdata, handles)
% hObject    handle to REMOVE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str=sprintf('File name');
str2=sprintf(' ');
set(handles.DataName, 'String', str);
set(handles.DATALIST, 'String', str2);


%% Make Paradigm
% --- Executes on button press in MakeParadigm.
function MakeParadigm_Callback(hObject, eventdata, handles)
% hObject    handle to MakeParadigm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str={'Motor Imagery','P300 speller','SSVEP'};
[Selection,ok] = listdlg('PromptString','Select an experiment type:',...
    'SelectionMode','single','ListString',str)
switch Selection
    case 1 % MI
        paradigm_MI
    case 2 % speller
        paradigm_p300
    case 3 % ssvep
end

% --- Executes during object creation, after setting all properties.
function MakeParadigm_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MakeParadigm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



%% Push button MI data analysis
% --- Executes on button press in MI.
function MI_Callback(hObject, eventdata, handles)
% hObject    handle to MI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.Preprocessing,'visible','on');
set(handles.FeatureExtraction,'visible','on');
set(handles.Classifier,'visible','on');
set(handles.Results,'visible','on');
set(handles.visualization,'visible','on');
set(handles.Introduction,'visible','off');
%% Push button SSVEP data analysis
% --- Executes on button press in SSVEP.
function SSVEP_Callback(hObject, eventdata, handles)
% hObject    handle to SSVEP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.Introduction,'visible','off');


%% Push button ERP data analysis
% --- Executes on button press in ERP.
function ERP_Callback(hObject, eventdata, handles)
% hObject    handle to ERP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.Introduction,'visible','off');

%% Push button Raw data analysis (Visualization)
% --- Executes on button press in RawData.
function RawData_Callback(hObject, eventdata, handles)
% hObject    handle to RawData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

RawData = handles.EEG;
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({RawData.data, RawData.marker, RawData.info}, field);

sz = get(0,'ScreenSize');     % get screen resolution
set(figure,'Position',[0,0,sz(3),sz(4)],'NumberTitle','off','Name','OpenBMI_DATA');
    %set(GUI.FIG,'Units', 'Normalized', 'Position',[0.05,0,0.9,0.85],'NumberTitle','off','Name','EEGDATA');
axes( 'Position',[0.1, 0.1, 0.65 ,0.85]);

status.epoch =1;
RawData.marker.fs = CNT.fs;
RawData.marker.pos = RawData.marker.t;
status.mrk = RawData.marker;

status.chan = CNT.chan;
status.initChan = CNT.chan;
status.CNT = 1;
status.Ival = [0 5000];

[status.scaling, status.scaleFactor]= opt_getScalingFactor(CNT , status.CNT);   % determine scaling factor for visualization
status.muVScale = opt_setMuVScale(CNT , status.CNT);                        % get the maximal value for the muV-bar
visual_updateShownEEG(CNT, status);
%% Pre-processing step
% --- Executes during object creation, after setting all properties.
function Preprocessing_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Preprocessing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object deletion, before destroying properties.
function Preprocessing_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to Preprocessing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


%% Select the Band-pass filter [-]Hz
function frequencyBand1_Callback(hObject, eventdata, handles)
% hObject    handle to frequencyBand1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of frequencyBand1 as text
%        str2double(get(hObject,'String')) returns contents of frequencyBand1 as a double
% global frequencyBand1
% frequencyBand1 = str2double(get(hObject,'String'));
% 
% --- Executes during object creation, after setting all properties.
function frequencyBand1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to frequencyBand1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% 
% 
function frequencyBand2_Callback(hObject, eventdata, handles)
% hObject    handle to frequencyBand2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of frequencyBand2 as text
%        str2double(get(hObject,'String')) returns contents of frequencyBand2 as a double
% global frequencyBand2
% frequencyBand2 = str2double(get(hObject,'String'));

% 
% --- Executes during object creation, after setting all properties.
function frequencyBand2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to frequencyBand2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%% Select the Spatial filter
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


%% Select the time interval [~]ms
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

%% Active prep-clear button
% --- Executes on button press in PrepOK.
function PrepOK_Callback(hObject, eventdata, handles)
% hObject    handle to PrepOK (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% frequencyBand_1 = str2double(get(handles.frequencyBand1,'String'));
% frequencyBand_2 = str2double(get(handles.frequencyBand2,'String'));
% 
% TimeInterval_1 = str2double(get(handles.TimeInterval1,'String'));
% TimeInterval_2 = str2double(get(handles.TimeInterval2,'String'));

if isempty(get(handles.frequencyBand1,'String'))== 1 || isempty(get(handles.frequencyBand2,'String'))==1
    warndlg(sprintf('Please input the frequency band'));
elseif isempty(get(handles.TimeInterval1,'String'))==1 || isempty(get(handles.TimeInterval2,'String'))==1
    warndlg(sprintf('Please input the time interval'));
else
    set(handles.prepClear,'Enable','on');
end



%% process the pre-processing step
% --- Executes on button press in prepClear.
function prepClear_Callback(hObject, eventdata, handles)
% hObject    handle to prepClear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


%--------------------------------------------------------------------------
% Load update data
global CNT SMT
DataEEG = handles.EEG;
% Setting the data
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({DataEEG.data, DataEEG.marker, DataEEG.info}, field);
%--------------------------------------------------------------------------
% Load the selected frequency band
frequencyBand_1 = str2double(get(handles.frequencyBand1,'String'));
frequencyBand_2 = str2double(get(handles.frequencyBand2,'String'));
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

handles.CNT = CNT;
handles.SMT = SMT;
guidata(hObject, handles)
msgbox('Complete the pre-processing step');


%% Select the Feature Extraction method

% --- Executes on selection change in selectFeatureExtraction.
function selectFeatureExtraction_Callback(hObject, eventdata, handles)
% hObject    handle to selectFeatureExtraction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectFeatureExtraction contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectFeatureExtraction

% --- Executes during object creation, after setting all properties.
function selectFeatureExtraction_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectFeatureExtraction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



%% Active Feature extraction button
% --- Executes on button press in FE_OK.
function FE_OK_Callback(hObject, eventdata, handles)
% hObject    handle to FE_OK (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FE = cellstr(get(handles.selectFeatureExtraction,'String'));
SelectFE = FE{get(handles.selectFeatureExtraction,'Value')};
if strcmp(SelectFE, 'None')==1
    warndlg(sprintf('Please select the feature extraction method'));
else
    set(handles.FEClear,'Enable','on');
end

%% Process Feature extraction
% --- Executes on button press in FEClear.
function FEClear_Callback(hObject, eventdata, handles)
% hObject    handle to FEClear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global FeatureSMT SelectFE
DataSMT = handles.SMT;

FE = cellstr(get(handles.selectFeatureExtraction,'String'));
SelectFE = FE{get(handles.selectFeatureExtraction,'Value')};

switch SelectFE
    case 'Log-variance'
        FeatureSMT=func_featureExtraction(DataSMT, {'feature','logvar'});
    case 'Common Spatial Pattern'
        [DataSMT, CSP_W, CSP_D]=func_csp(DataSMT,{'nPatterns', [3]});
        FeatureSMT=func_featureExtraction(DataSMT, {'feature','logvar'});
end

handles.FeatureSMT = FeatureSMT;
handles.SelectFE=SelectFE;
guidata(hObject, handles)
msgbox('Complete the feature extraction step');


%% Select the classifier & evatuation method
% --- Executes on selection change in selectClassifier.
function selectClassifier_Callback(hObject, eventdata, handles)
% hObject    handle to selectClassifier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectClassifier contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectClassifier

% --- Executes during object creation, after setting all properties.
function selectClassifier_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectClassifier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in selectEvaluation.
function selectEvaluation_Callback(hObject, eventdata, handles)
% hObject    handle to selectEvaluation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectEvaluation contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectEvaluation

% --- Executes during object creation, after setting all properties.
function selectEvaluation_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectEvaluation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



%% Process the classifier output
% --- Executes on button press in CLSClear.
function CLSClear_Callback(hObject, eventdata, handles)
% hObject    handle to CLSClear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global FeatureSMT CNT SMT SelectFE
FeatureSMT = handles.FeatureSMT;
CNT = handles.CNT;
SMT = handles.SMT;
SelectFE = handles.SelectFE;
% feature extraction 방법도 불러오는 것 필요. 지금은 log-var밖에 없어서

% Classification. Load the type of classifier
Classifier = cellstr(get(handles.selectClassifier,'String'));
SelectClassifier = Classifier{get(handles.selectClassifier,'Value')};
if strcmp (SelectClassifier,'Linear Discriminant Analysis (LDA)')
    clf_param=func_train(FeatureSMT,{'classifier','LDA'});
elseif strcmp (SelectClassifier,'Suppot Vector Machine (SVM)')
    
end

% Evaluation. Load the type of evaluation
Evaluation = cellstr(get(handles.selectEvaluation,'String'));
SelectEvaluation = Evaluation{get(handles.selectEvaluation,'Value')};

% feature extraction method
switch SelectFE
    case 'Log-variance'
        feature='logvar';
    case 'Common Spatial Pattern'
        feature='csp';
end
% classification method
if strcmp (SelectClassifier,'Linear Discriminant Analysis (LDA)')
    classifier='LDA';
elseif strcmp (SelectClassifier,'Suppot Vector Machine (SVM)')
    classifier='SVM';
end
% evaluation method
if strcmp (SelectEvaluation,'k-fold cross-validation')
    prompt={'Enter the number of folds:'};
    defaultans = {'5'};
    Kval=inputdlg(prompt,'K-fold cross validation',[1 70],defaultans);
    PerformanceEvaluation='KFold';
elseif strcmp(SelectEvaluation,'Leave-one-out CV')
    
else
    
end

CV.var.fv=feature;
CV.var.classf=classifier;
CV.var.evaluation=PerformanceEvaluation;
CV.var.k=str2num(cell2mat(Kval)); %temp
switch(feature)
    case 'csp'
        CV.var.fv='logvar';
        CV.train={
            '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
            'FT=func_featureExtraction(SMT, {"feature",fv})'
            '[CF_PARAM]=func_train(FT,{"classifier",classf})'
            };
        CV.test={
            'SMT=func_projection(SMT, CSP_W)'
            'FT=func_featureExtraction(SMT, {"feature",fv})'
            '[cf_out]=func_predict(FT, CF_PARAM)'
            };
        CV.option={
            'evaluation' , 'k'
            };
    case 'logvar'
        CV.train={
            'FT=func_featureExtraction(SMT, {"feature",fv})'
            '[CF_PARAM]=func_train(FT,{"classifier",classf})'
            };
        CV.test={
            'FT=func_featureExtraction(SMT, {"feature",fv})'
            '[cf_out]=func_predict(FT, CF_PARAM)'
            };
        CV.option={
            'evaluation' , 'k'
            };
end
[loss]=eval_crossValidation(SMT, CV);
set(handles.ACC, 'String', sprintf('%.2f',(1-loss)*100));

handles.Classifier = clf_param;
msgbox('Complete the classification');

%% Active classifier button
% --- Executes on button press in CLS_OK.
function CLS_OK_Callback(hObject, eventdata, handles)
% hObject    handle to CLS_OK (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Classifier = cellstr(get(handles.selectClassifier,'String'));
SelectClassifier = Classifier{get(handles.selectClassifier,'Value')};

Evaluation = cellstr(get(handles.selectEvaluation,'String'));
SelectEvaluation = Evaluation{get(handles.selectEvaluation,'Value')};

if strcmp(SelectClassifier, 'None')== 1
    warndlg(sprintf('Please select the classifier method'));
elseif strcmp(SelectEvaluation, 'None') == 1
    warndlg(sprintf('You do not select the evaluation methods \n If you want to evaluate classifier, please select the evaluation methods'));
    set(handles.CLSClear,'Enable','on');
else
    set(handles.CLSClear,'Enable','on');
end



%% Show the classification accuracy
% --- Executes during object creation, after setting all properties.
function ACC_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ACC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in visual_scalpPlot.
function visual_scalpPlot_Callback(hObject, eventdata, handles)
% hObject    handle to visual_scalpPlot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global ScalpEEG
global ScalpCNT
global ScalpSMT
ScalpEEG = handles.EEG;
ScalpCNT = handles.CNT;
ScalpSMT = handles.SMT;

prompt={'Select time interval:','Trial:'};
defaultans = {'[100 200]','5'};
c=inputdlg(prompt,'time Interval',[1 70],defaultans);
c{1}=strrep(c{1},'[','');

c{1}=strrep(c{1},']','');
t = strsplit(c{1} , ' ');
timeInterval1= str2num(t{1});
timeInterval2= str2num(t{2});
trial = str2num(c{2});
if timeInterval1 >= size(ScalpSMT.x,1)
    warndlg(sprintf('Please select initial time interval again'));
elseif timeInterval2 >= size(ScalpSMT.x,1)
    warndlg(sprintf('Please select end time interval again'));
else
    w = mean(ScalpSMT.x(timeInterval1:timeInterval2,trial,:),1);
    w = squeeze(w)';
    visual_TopoPlot(ScalpSMT, w, ScalpCNT);
end


% --- Executes during object creation, after setting all properties.
function visual_scalpPlot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to visual_scalpPlot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in visual_powerSpect.
function visual_powerSpect_Callback(hObject, eventdata, handles)
% hObject    handle to visual_powerSpect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in visual_ERSP.
function visual_ERSP_Callback(hObject, eventdata, handles)
% hObject    handle to visual_ERSP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in visual_contiScalpPattern.
function visual_contiScalpPattern_Callback(hObject, eventdata, handles)
% hObject    handle to visual_contiScalpPattern (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global ScalpEEG
global ScalpCNT
global ScalpSMT
global MNT
ScalpEEG = handles.EEG;
ScalpCNT = handles.CNT;
ScalpSMT = handles.SMT;

MNT = opt_getMontage(ScalpSMT);
center = [0 0];                   
theta = linspace(0,2*pi,360);  
x = cos(theta)+center(1);  
y = sin(theta)+center(2);  
oldUnit = get(gcf,'units');
set(gcf,'units','normalized');
H = struct('ax', gca);
set(gcf,'CurrentAxes',H.ax);
tic
xe_org = MNT.x';
ye_org = MNT.y';
resolution = 100;
maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));
xx = linspace(-maxrad, maxrad, resolution);
yy = linspace(-maxrad, maxrad, resolution)';

tmpfig2 = prep_envelope(ScalpCNT);
s_1=1;
b_size1=100;
s_size=10;
for i = 1:10000
%     %figure 1
% %     subplot(1,2,1)
% %     tm2.x=flt_CNT.x(s_1:s_1+b_size1,:);
% %     tm2.fs = 100;
% %     tmp3 = prep_envelope(tm2);
% %     tm1=mean(tmp3.x(b_size1-50:b_size1,:));
% %     visual_topoplot(tm1, xe_org, ye_org, xx, yy);
% %     drawnow;
%     % figure 2
% 
% %     figure(1)
% %     subplot(1,2,2)
    tmpfig1=tmpfig2.x(s_1:s_1+b_size1,:);
    tmpfig3=mean(tmpfig1(b_size1-50:b_size1,:));
    visual_contiScaplPlot(tmpfig3, xe_org, ye_org, xx, yy ,x,y);
    
    drawnow;
    s_1=s_1+s_size;
end



% --- Executes during object creation, after setting all properties.
function visual_powerSpect_CreateFcn(hObject, eventdata, handles)
% hObject    handle to visual_powerSpect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function visual_ERSP_CreateFcn(hObject, eventdata, handles)
% hObject    handle to visual_ERSP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function visual_contiScalpPattern_CreateFcn(hObject, eventdata, handles)
% hObject    handle to visual_contiScalpPattern (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function Results_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Results (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function visualization_CreateFcn(hObject, eventdata, handles)
% hObject    handle to visualization (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function Introduction_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Introduction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function LOGO_CreateFcn_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LOGO (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate LOGO
