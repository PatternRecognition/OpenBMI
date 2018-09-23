function varargout = select_class(varargin)
% SELECT_CLASS MATLAB code for select_class.fig
%      SELECT_CLASS, by itself, creates a new SELECT_CLASS or raises the existing
%      singleton*.
%
%      H = SELECT_CLASS returns the handle to a new SELECT_CLASS or the handle to
%      the existing singleton*.
%
%      SELECT_CLASS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SELECT_CLASS.M with the given input arguments.
%
%      SELECT_CLASS('Property','Value',...) creates a new SELECT_CLASS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before select_class_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to select_class_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help select_class

% Last Modified by GUIDE v2.5 14-Nov-2017 21:19:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @select_class_OpeningFcn, ...
                   'gui_OutputFcn',  @select_class_OutputFcn, ...
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


% --- Executes just before select_class is made visible.
function select_class_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to select_class (see VARARGIN)

% Choose default command line output for select_class
handles.output = hObject;

if isempty(varargin)
    handles.init_class={'target','non-target'};
else
    handles.smt_class=varargin{1};
    handles.init_class=varargin{2};
end

set(handles.ck_class1,'Visible','off');
set(handles.ck_class2,'Visible','off');
set(handles.ck_class3,'Visible','off');

% Update handles structure
guidata(hObject, handles);
initialize_gui(hObject, handles, false);
% UIWAIT makes select_class wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = select_class_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
delete(handles.figure1);


function initialize_gui(hObject, handles, isreset)
% Update handles structure
for i=1:length(handles.smt_class)
    eval(sprintf('set(handles.ck_class%d,''Visible'',''on'');',i));
    eval(sprintf('set(handles.ck_class%d,''String'',handles.smt_class{i});',i));
end
guidata(hObject, handles);
RESET(hObject,handles);




function RESET(hObject, handles, isreset)
% Update handles structure
class=handles.init_class;

value_logic=ismember(handles.smt_class, handles.init_class);
for i=1:length(handles.smt_class)
    eval(sprintf('set(handles.ck_class%d,''Value'',value_logic(i));',i));
end

guidata(hObject, handles);

% --- Executes on button press in ck_class1.
function ck_class1_Callback(hObject, eventdata, handles)
% hObject    handle to ck_class1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ck_class1


% --- Executes on button press in ck_class2.
function ck_class2_Callback(hObject, eventdata, handles)
% hObject    handle to ck_class2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ck_class2


% --- Executes on button press in ck_class3.
function ck_class3_Callback(hObject, eventdata, handles)
% hObject    handle to ck_class3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ck_class3


% --- Executes on button press in apply_btn.
function apply_btn_Callback(hObject, eventdata, handles)
% hObject    handle to apply_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.selected_class_logic=[];
for i=1:length(handles.smt_class)
    eval(sprintf('handles.selected_class_logic=[handles.selected_class_logic handles.ck_class%d.Value];',i));
end
handles.selected_class=handles.smt_class(find(handles.selected_class_logic));
handles.output=handles.selected_class;
guidata(hObject,handles);
uiresume(handles.figure1);


% --- Executes on button press in reset_btn.
function reset_btn_Callback(hObject, eventdata, handles)
% hObject    handle to reset_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
RESET(hObject, handles)
