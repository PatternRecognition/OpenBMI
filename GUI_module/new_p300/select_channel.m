function varargout = select_channel(varargin)
% SELECT_CHANNEL MATLAB code for select_channel.fig
%      SELECT_CHANNEL, by itself, creates a new SELECT_CHANNEL or raises the existing
%      singleton*.
%
%      H = SELECT_CHANNEL returns the handle to a new SELECT_CHANNEL or the handle to
%      the existing singleton*.
%
%      SELECT_CHANNEL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SELECT_CHANNEL.M with the given input arguments.
%
%      SELECT_CHANNEL('Property','Value',...) creates a new SELECT_CHANNEL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before select_channel_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to select_channel_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help select_channel

% Last Modified by GUIDE v2.5 08-Aug-2017 16:55:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @select_channel_OpeningFcn, ...
                   'gui_OutputFcn',  @select_channel_OutputFcn, ...
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


% --- Executes just before select_channel is made visible.
function select_channel_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to select_channel (see VARARGIN)

% Choose default command line output for select_channel
handles.output = hObject;
global state
chan = {...
    '','','','Fp1','','Fpz','', 'Fp2','','','';
    '','','','','AFp1','','AFp2', '','','','';
    'F9','AF7','', 'AF3','','AFz','','AF4', '','AF8','F10';
    '', '','AFF5h','','AFF1h', '', 'AFF2h', '', 'AFF6h','','';
    '','F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8','';
    'FFT9h','FFT7h','FFC5h','FFC3h','FFC1h','','FFC2h','FFC4h','FFC6h','FFT8h','FFT10h';
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10';
    'FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h';
    '', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8','';
    '', 'TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', '', 'CCP2h', 'CCP5h', 'CCP6h', 'TTP8h', '';
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10';
    'TPP9h', 'TPP7h', 'CPP5h', 'CPP3h', 'CPP1h', '', 'CPP2h', 'CPP4h', 'CPP6h', 'TPP8h', 'TPP10h';
    'P9','P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8','P10';
    'PPO9h', '', 'PPO5h', '', 'PPO1h', '', 'PPO2h', '', 'PPO6h', '', 'PPO10h';
    '','PO7','','PO3', '', 'POz', '', 'PO4','','PO8','';
    '','PO9','POO9h','O1','POO1','','POO2','O2','POO10h','PO10','';
    '','','','l1', 'Ol1h', 'Oz', 'Olh2', 'l2','','','';
    '','','','', '', 'lz', '', '','','',''};

a = cell(size(chan,1)+1, size(chan,2));
j = size(chan,2);
for i = 1: size(state.clab, 2)
    [row, col]= find(strcmp(chan,state.clab{i}));
    if(isempty(row)&&isempty(col))
        a{19, j} = sprintf('%s (%d)',state.clab{i},i);
        j = j-1;
    else
        a{row, col} = sprintf('%s (%d)',state.clab{i},i);
    end
end
set(handles.uitable2,'Data',a);
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes select_channel wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = select_channel_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
