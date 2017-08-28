function varargout = p300_DataAnalysis(varargin)
% P300_DATAANALYSIS MATLAB code for p300_DataAnalysis.fig
%      P300_DATAANALYSIS, by itself, creates a new P300_DATAANALYSIS or raises the existing
%      singleton*.
%
%      H = P300_DATAANALYSIS returns the handle to a new P300_DATAANALYSIS or the handle to
%      the existing singleton*.
%
%      P300_DATAANALYSIS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in P300_DATAANALYSIS.M with the given input arguments.
%
%      P300_DATAANALYSIS('Property','Value',...) creates a new P300_DATAANALYSIS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before p300_DataAnalysis_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to p300_DataAnalysis_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help p300_DataAnalysis

% Last Modified by GUIDE v2.5 23-Aug-2017 13:37:59

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @p300_DataAnalysis_OpeningFcn, ...
                   'gui_OutputFcn',  @p300_DataAnalysis_OutputFcn, ...
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


% --- Executes just before p300_DataAnalysis is made visible.
function p300_DataAnalysis_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to p300_DataAnalysis (see VARARGIN)
set(gcf,'units','points','position',[500 50 490 625]);
set(handles.TimeIvalFirst ,'String','-200');
set(handles.TimeIvalEnd ,'String','800');
set(handles.baselineTimeFirst ,'String','-200');
set(handles.baselineTimeEnd  ,'String','0');
set(handles.selectedTimeFirst  ,'String','0');
set(handles.selectedTimeEnd  ,'String','800');
set(handles.selectedFreqFirst  ,'String','0.5');
set(handles.selectedFreqEnd  ,'String','40');
set(handles.chanSel  ,'String','all');
set(handles.numFeature  ,'String','10');
set(handles.REMOVE,'visible','off');
set(handles.artifactReject,'String','300');
set(handles.artifactChan,'String','Cz, C1, C2');
set(handles.tcpiptxt,'String','12300');
set(handles.loadtrainpamter,'visible','off');
set(handles.closebtn,'visible','off');
% Choose default command line output for p300_DataAnalysis
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
% UIWAIT makes p300_DataAnalysis wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function initialize_gui(fig_handle, handles, isreset)
% If the metricdata field is present and the reset flag is false, it means
% we are we are just re-initializing a GUI by calling it from the cmd line
% while it is up. So, bail out as we dont want to reset the data.
if isfield(handles, 'metricdata') && ~isreset
    return;
end
RESET(handles, true);
% Update handles structure
guidata(handles.figure1, handles);

% --- Outputs from this function are returned to the command line.
function varargout = p300_DataAnalysis_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Connect.
function Connect_Callback(hObject, eventdata, handles)
% hObject    handle to Connect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global aclf_param ach_idx aTimeIval aselectedTime anumFeature abaselineTime aselectedFreq state
if get(handles.loadradio,'Value')
chanSell = get(handles.chanSel,'String');
    if isequal(chanSell,'all')  
        [output, OTHERS, STRMATRIX] = filt_EEG_CHANNEL(state.clab);
        chanSel= cell2mat(output(:,1))'; 
    else
        chanSel = str2double(strsplit(get(handles.chanSel,'String'), ' ,'));
    end    
        TimeIvalFirst = str2double(get(handles.TimeIvalFirst,'String'));
        TimeIvalEnd = str2double(get(handles.TimeIvalEnd,'String'));
        baselineTimeFirst = str2double(get(handles.baselineTimeFirst,'String'));
        baselineTimeEnd = str2double(get(handles.baselineTimeEnd,'String'));
        selectedTimeFirst = str2double(get(handles.selectedTimeFirst,'String'));
        selectedTimeEnd = str2double(get(handles.selectedTimeEnd,'String'));
        numFeature = str2double(get(handles.numFeature,'String'));
        selectedFreqFirst = str2double(get(handles.selectedFreqFirst,'String'));
        selectedFreqEnd = str2double(get(handles.selectedFreqEnd,'String'));
        artifactReject = str2double(get(handles.artifactReject,'String'));
        artifactChan = get(handles.artifactChan,'String');
        TimeIval = [TimeIvalFirst TimeIvalEnd];
        baselineTime = [baselineTimeFirst baselineTimeEnd];
        selectedTime = [selectedTimeFirst selectedTimeEnd];
        selectedFreq = [selectedFreqFirst selectedFreqEnd];
        aclf_param = handles.clf_parammm;
        aTimeIval = TimeIval;
        aselectedTime = selectedTime;
        anumFeature = numFeature;
        abaselineTime = baselineTime;
        aselectedFreq = selectedFreq;
        ach_idx = chanSel;
        set(handles.notiontxt,'String','Connecting with paradigm...'); drawnow;
    if get(handles.rcbtn, 'Value')
        p300_online_rc(handles.axes1, handles.text51, {'segTime',aTimeIval;'baseTime',abaselineTime;'selTime',aselectedTime;'nFeature',anumFeature;'channel',ach_idx; 'clf_param',aclf_param; 'selectedFreq', aselectedFreq} );
    else
        p300_online(handles.axes1, handles.text51, {'segTime',aTimeIval;'baseTime',abaselineTime;'selTime',aselectedTime;'nFeature',anumFeature;'channel',ach_idx; 'clf_param',aclf_param; 'selectedFreq', aselectedFreq} );  
    end
else
%%        
    chanSell = get(handles.chanSel,'String');
    if isequal(chanSell,'all')  
        [output, OTHERS, STRMATRIX] = filt_EEG_CHANNEL(state.clab);
        chanSel= cell2mat(output(:,1))'; 
    else
        chanSel = str2double(strsplit(get(handles.chanSel,'String'), ' ,'));
    end        
        aTimeIval = handles.TimeIvall ;
        aselectedTime = handles.selectedTimee; 
        anumFeature = handles.numFeaturee ;
        abaselineTime = handles.baselineTimee;
        aselectedFreq = handles.selectedFreqq ;
        handles.fvv ;
        ach_idx = chanSel;
        aclf_param = handles.clf_paramm;
        set(handles.notiontxt,'String','Connecting with paradigm...'); drawnow;
    if get(handles.rcbtn, 'Value')
        p300_online_rc(handles.axes1, handles.text51, {'segTime',aTimeIval;'baseTime',abaselineTime;'selTime',aselectedTime;'nFeature',anumFeature;'channel',ach_idx; 'clf_param',aclf_param; 'selectedFreq', aselectedFreq} );
    else
        p300_online(handles.axes1, handles.text51, {'segTime',aTimeIval;'baseTime',abaselineTime;'selTime',aselectedTime;'nFeature',anumFeature;'channel',ach_idx; 'clf_param',aclf_param; 'selectedFreq', aselectedFreq} );  
    end
end

set(handles.axes1,'XTickLabel', '');
set(handles.axes1,'YTickLabel', '');
delete(get(handles.axes1,'Children'));

set(handles.notiontxt,'String','Done all process...');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ADD.
function ADD_Callback(hObject, eventdata, handles)
% hObject    handle to ADD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(handles.loadradio,'Value')
global check_load
check_load=true;
[file,path]=uigetfile('*.mat','Load parameters');
p300Param=load(sprintf('%s%s',path,file));
clf_param = p300Param.CF_PARAM;
handles.clf_parammm=clf_param;
weight= size(clf_param.cf_param.w);
bias = clf_param.cf_param.b;
classifier = clf_param.classifier;
str2=sprintf('Weight: %d x %d \nClassifier: %s\nbias: %f',weight,classifier,bias);
set(handles.text46, 'String', str2);
else
global EEG check_add
set(handles.notiontxt,'String','processing EEG data...'); drawnow;
[file,path]=uigetfile('*.eeg','Load EEG file (.eeg)');
[file]=strsplit(file,'.eeg');
prompt={'Enter trigger number and corresponding class name:','Enter sampling frequency:'};
defaultans = {'{''1'',''Target'';''2'',''Non-target''}','100'};
c=inputdlg(prompt,'Marker',[1 70],defaultans);
cls=c{1};
fs=str2double(c{2});
[EEG.data, EEG.marker, EEG.info]=Load_EEG([path,file{1}],{'device','brainVision';'marker',eval(cls);'fs',fs});
dataSize=size(EEG.data.x);
marker=str2double(EEG.marker.class(:,1)');
fs=EEG.info.fs;
className=EEG.marker.class(:,2)';
numberStimulus=size(EEG.marker.y_dec,2);
str=sprintf('Data size: %.0fx%.0f\nNumber of channels: %d\n',dataSize(1),dataSize(2),dataSize(2));
str2=sprintf('Frequency: %d Hz\nClass: %s\nNumber of trials: %d\n',fs,strjoin(className),numberStimulus);
set(handles.text46, 'String', cat(2,str,str2));
% set(handles.notiontxt,'FontSize',12);
set(handles.notiontxt,'String','Done and process next step (feature, classifier)'); 
check_add=true;
handles.concat = cat(2,str,str2);
handles.EEG = EEG;
end

guidata(hObject, handles)

% --- Executes on button press in REMOVE.
function REMOVE_Callback(hObject, eventdata, handles)
% hObject    handle to REMOVE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function TimeIvalFirst_Callback(hObject, eventdata, handles)
% hObject    handle to TimeIvalFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TimeIvalFirst as text
%        str2double(get(hObject,'String')) returns contents of TimeIvalFirst as a double
input = get(handles.TimeIvalFirst,'string');
if regexp(input, '[^0-9.,-]')
    set(handles.notiontxt,'String','TimeIval first input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.TimeIvalEnd,'string');
if regexp(input, '[^0-9.,-]')
    set(handles.notiontxt,'String','TimeIval end input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.chanSel,'string');
if isequal(input, 'all');
    set(handles.notiontxt,'String','Set');
    return;  
else 
    if regexp(input, '[^0-9., ;]')
            set(handles.notiontxt,'String','chan selection input format should be checked');
    else
            set(handles.notiontxt,'String','Set');
    end    
end

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


% --- Executes on button press in makeClassifier.
function makeClassifier_Callback(hObject, eventdata, handles)
% hObject    handle to makeClassifier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.notiontxt,'String','Classifying data...'); drawnow;
fv = handles.fvv;
ch_idx = handles.ch_idxx; % classification
[clf_param] = func_train(fv,{'classifier','LDA'});

weight= size(clf_param.cf_param.w);
bias = clf_param.cf_param.b;
classifier = clf_param.classifier;
str2=sprintf('Feature: %d x %d \nWeight: %d x %d \nClassifier: %s\nbias: %f',size(fv.x),weight,classifier,bias);
c = handles.concat;
cc = cat(2,c,str2);
set(handles.text46, 'String', cc);
handles.clf_paramm=clf_param;
%%%%%%%%%%%%%%%%%
% EEG = handles.EEG;
% field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
% CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% 
% selectedFreqFirst = str2double(get(handles.selectedFreqFirst,'String'));
% selectedFreqEnd = str2double(get(handles.selectedFreqEnd,'String'));
% CV.var.band = [selectedFreqFirst selectedFreqEnd];
% selectedTimeFirst = str2double(get(handles.selectedTimeFirst,'String'));
% selectedTimeEnd = str2double(get(handles.selectedTimeEnd,'String'));
% CV.var.interval= [selectedTimeFirst selectedTimeEnd];
% 
% CV.prep={ % commoly applied to training and test data before data split
%     'CNT=prep_filter(CNT, {"frequency", band})'
%     'SMT=prep_segmentation(CNT, {"interval", interval})'
%     };
% CV.train={
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
%     };
% CV.test={
%     'FT=func_featureExtraction(SMT, {"feature","logvar"})'
%     '[cf_out]=func_predict(FT, CF_PARAM)'
%     };
% CV.option={
% 'KFold','10'
% % 'leaveout'
% };
% 
% [loss]=eval_crossValidation(CNT, CV); % input : eeg, or eeg_epo
% set(handles.text51,'String',1-loss); 
%%%%%%%%%%%%%%%%%
set(handles.notiontxt,'String','Done data classification...'); 

guidata(hObject, handles)


function baselineTimeFirst_Callback(hObject, eventdata, handles)
% hObject    handle to baselineTimeFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baselineTimeFirst as text
%        str2double(get(hObject,'String')) returns contents of baselineTimeFirst as a double
input = get(handles.baselineTimeFirst,'string');
if regexp(input, '[^0-9.,-]')
    set(handles.notiontxt,'String','baselineTime first input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.baselineTimeEnd,'string');
if regexp(input, '[^0-9.,-]')
    set(handles.notiontxt,'String','baselineTime input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.selectedTimeFirst,'string');
if regexp(input, '[^0-9.,-]')
    set(handles.notiontxt,'String','selected Time Firstinput format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.selectedTimeEnd,'string');
if regexp(input, '[^0-9.,-]')
    set(handles.notiontxt,'String','selected Time End input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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
input = get(handles.numFeature,'string');
if regexp(input, '[^0-9.,-]')
    set(handles.notiontxt,'String','num Feature input format should be checked');
else
        set(handles.notiontxt,'String','Set');
end

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


% --- Executes on button press in saveparameter.
function saveparameter_Callback(hObject, eventdata, handles)
% hObject    handle to saveparameter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
time = handles.TimeIvall;
selecttime= handles.selectedTimee;
nuFeature = handles.numFeaturee;
baseline = handles.baselineTimee;
freq = handles.selectedFreqq;
features = handles.fvv ;
chan_idx = handles.ch_idxx;
param = handles.clf_paramm;
p300Param.timeival = time ;
p300Param.selectedtime= selecttime;
p300Param.numfeatures = nuFeature;
p300Param.baselinetime = baseline;
p300Param.ch_idx = chan_idx ;
p300Param.selectedFreq = freq;
p300Param.fv =features;
p300Param.clf_param = param;

uisave('p300Param','p300Param');
set(handles.notiontxt,'String','Done save parameters...');

% --- Executes on button press in loadtrainpamter.
function loadtrainpamter_Callback(hObject, eventdata, handles)
% hObject    handle to loadtrainpamter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function artifactReject_Callback(hObject, eventdata, handles)
% hObject    handle to artifactReject (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of artifactReject as text
%        str2double(get(hObject,'String')) returns contents of artifactReject as a double


% --- Executes during object creation, after setting all properties.
function artifactReject_CreateFcn(hObject, eventdata, handles)
% hObject    handle to artifactReject (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in featureExtract.
function featureExtract_Callback(hObject, eventdata, handles)
% hObject    handle to featureExtract (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global EEG TimeIval baselineTime selectedTime selectedFreq state chanSel
EEG = handles.EEG;
set(handles.notiontxt,'String','Extracting features...'); drawnow;

chanSell = get(handles.chanSel,'String');
if isequal(chanSell,'all')  
    [output, OTHERS, STRMATRIX] = filt_EEG_CHANNEL(EEG.info.chan);
    chanSel= cell2mat(output(:,1))'; 
else
    chanSel = str2double(strsplit(get(handles.chanSel,'String'), ' ,'));
end

TimeIvalFirst = str2double(get(handles.TimeIvalFirst,'String'));
TimeIvalEnd = str2double(get(handles.TimeIvalEnd,'String'));
baselineTimeFirst = str2double(get(handles.baselineTimeFirst,'String'));
baselineTimeEnd = str2double(get(handles.baselineTimeEnd,'String'));
selectedTimeFirst = str2double(get(handles.selectedTimeFirst,'String'));
selectedTimeEnd = str2double(get(handles.selectedTimeEnd,'String'));
numFeature = str2double(get(handles.numFeature,'String'));
selectedFreqFirst = str2double(get(handles.selectedFreqFirst,'String'));
selectedFreqEnd = str2double(get(handles.selectedFreqEnd,'String'));
artifactReject = str2double(get(handles.artifactReject,'String'));
artifactChan = get(handles.artifactChan,'String');

TimeIval = [TimeIvalFirst TimeIvalEnd];
baselineTime = [baselineTimeFirst baselineTimeEnd];
selectedTime = [selectedTimeFirst selectedTimeEnd];
selectedFreq = [selectedFreqFirst selectedFreqEnd];

% feature extraction
% [fv,ch_idx] = p300_featureExtraction(EEG,{'segTime',TimeIval;'baseTime',baselineTime;'selTime',selectedTime;'nFeature',numFeature;'channel',chanSel;'Freq',selectedFreq;'Artifact_thres',artifactReject;'Artifact_chan',artifactChan});
[fv,ch_idx] = p300_featureExtraction(EEG,{'segTime',TimeIval;'baseTime',baselineTime;'selTime',selectedTime;'nFeature',numFeature;'channel',chanSel;'Freq',selectedFreq});
set(handles.notiontxt,'String','Donw feature extraction');
handles.TimeIvall = TimeIval;
handles.selectedTimee = selectedTime;
handles.numFeaturee= numFeature;
handles.baselineTimee = baselineTime;
handles.selectedFreqq = selectedFreq;
handles.fvv = fv;
handles.ch_idxx = ch_idx;

guidata(hObject, handles)

function selectedFreqFirst_Callback(hObject, eventdata, handles)
% hObject    handle to selectedFreqFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of selectedFreqFirst as text
%        str2double(get(hObject,'String')) returns contents of selectedFreqFirst as a double


% --- Executes during object creation, after setting all properties.
function selectedFreqFirst_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectedFreqFirst (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function selectedFreqEnd_Callback(hObject, eventdata, handles)
% hObject    handle to selectedFreqEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of selectedFreqEnd as text
%        str2double(get(hObject,'String')) returns contents of selectedFreqEnd as a double


% --- Executes during object creation, after setting all properties.
function selectedFreqEnd_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectedFreqEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function artifactChan_Callback(hObject, eventdata, handles)
% hObject    handle to artifactChan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of artifactChan as text
%        str2double(get(hObject,'String')) returns contents of artifactChan as a double


% --- Executes during object creation, after setting all properties.
function artifactChan_CreateFcn(hObject, eventdata, handles)
% hObject    handle to artifactChan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in makeradio.
function makeradio_Callback(hObject, eventdata, handles)
% hObject    handle to makeradio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.makeradio,'Value'))
    set(handles.loadradio,'Value',0)
end
set(handles.loadtrainpamter,'visible','off');
set(handles.featureExtract,'visible','on');
set(handles.makeClassifier,'visible','on');
% Hint: get(hObject,'Value') returns toggle state of makeradio
set(handles.notiontxt,'String','Make new classifier');

% --- Executes on button press in loadradio.
function loadradio_Callback(hObject, eventdata, handles)
% hObject    handle to loadradio (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.loadradio,'Value'))
    set(handles.makeradio,'Value',0)
end
set(handles.loadtrainpamter,'visible','off');
set(handles.featureExtract,'visible','off');
set(handles.makeClassifier,'visible','off');
set(handles.notiontxt,'String','Load previous classifier');

% global clf_param check_load
% check_load=true;
% [file,path]=uigetfile('*.mat','Load parameters');
% clf_param=load(sprintf('%s%s',path,file));
% handles.clf_parammm=clf_param;
% guidata(hObject, handles)
% Hint: get(hObject,'Value') returns toggle state of loadradio

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


% --- Executes on button press in checktcpip.
function checktcpip_Callback(hObject, eventdata, handles)
% hObject    handle to checktcpip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock state

if ~isempty(sock)
%     disp('sock is empty');
   if ~isequal(sock.status, 'closed')
    set(handles.notiontxt,'String','Already connected');
    return;
    end
end

bbci_acquire_bv('close');
tcpipp = str2double(get(handles.tcpiptxt,'String'));
param = struct;
state = bbci_acquire_bv('init',param);
sock = tcpip('localhost',tcpipp);  

try 
    fopen(sock);
    while(true) % 19 check
    fwrite(sock,19);    
    [~,~,marker,~] = bbci_acquire_bv(state);
        if marker == 19
            set(handles.checktcpip, 'string', 'OK'); 
            set(handles.closebtn,'visible','on'); 
                set(handles.notiontxt,'String','Connected');
            break;
        end
% set(handles.notiontxt,'String','Check connection of TCPIP, let`s start pardigm');
    end
catch 
    disp('Error: Check tcpip again');
end

% --- Executes on button press in resetbtn.
function resetbtn_Callback(hObject, eventdata, handles)
% hObject    handle to resetbtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
RESET(handles, false);

function RESET(handles, init)
if ~init
    a = [500 50 490 625];
else
    a = get(handles.figure1,'Position');
end

set(handles.figure1,'Position',a);
% set(gcf,'units','points','position',[500 200 490 625]);
set(handles.TimeIvalFirst ,'String','-200');
set(handles.TimeIvalEnd ,'String','800');
set(handles.baselineTimeFirst ,'String','-200');
set(handles.baselineTimeEnd  ,'String','0');
set(handles.selectedTimeFirst  ,'String','0');
set(handles.selectedTimeEnd  ,'String','800');
set(handles.selectedFreqFirst  ,'String','0.5');
set(handles.selectedFreqEnd  ,'String','40');
set(handles.chanSel  ,'String','9, 10');
set(handles.numFeature  ,'String','10');
set(handles.artifactReject,'String','300');
set(handles.artifactChan,'String','Cz, C1, C2');
set(handles.tcpiptxt,'String','12300');
set(handles.loadradio,'Value',0);
set(handles.makeradio,'Value',0);
set(handles.text46, 'String', '');
set(handles.closebtn,'visible','off'); 
 
global state sock
bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
disp('will close to connection');
try
    fclose(sock);
catch
end


% --- Executes on button press in chanbtn.
function chanbtn_Callback(hObject, eventdata, handles)
% hObject    handle to chanbtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global EEG aaa
if get(handles.makeradio,'Value')
    EEG = handles.EEG;
    aaa = EEG.info.chan;
    select_channel_ver;   
else
    select_channel;
end


% --- Executes on button press in closebtn.
function closebtn_Callback(hObject, eventdata, handles)
% hObject    handle to closebtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global sock
fclose(sock);
set(handles.checktcpip, 'string', 'check');
set(handles.notiontxt,'String','Disconnect tcpip connection');
set(handles.closebtn,'visible','off'); 


% --- Executes on button press in rcbtn.
function rcbtn_Callback(hObject, eventdata, handles)
% hObject    handle to rcbtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.rcbtn,'Value'))
    set(handles.randombtn,'Value',0)
    set(handles.facebtn,'Value',0)
end
% Hint: get(hObject,'Value') returns toggle state of rcbtn


% --- Executes on button press in randombtn.
function randombtn_Callback(hObject, eventdata, handles)
% hObject    handle to randombtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.randombtn,'Value'))
    set(handles.rcbtn,'Value',0)
    set(handles.facebtn,'Value',0)
end
% Hint: get(hObject,'Value') returns toggle state of randombtn


% --- Executes on button press in facebtn.
function facebtn_Callback(hObject, eventdata, handles)
% hObject    handle to facebtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(get(handles.facebtn,'Value'))
    set(handles.rcbtn,'Value',0)
    set(handles.randombtn,'Value',0)
end
% Hint: get(hObject,'Value') returns toggle state of facebtn
