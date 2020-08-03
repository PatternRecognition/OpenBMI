function bbci = gui_file_setup(bbci);
%GUI_FILE_SETUP OPENS A GUI TO LOAD FILES AND SPECIFY SOME DEFAULT 
%FOR THE TRAINING OF A CLASSIFIER
%
% usage:
%    bbci = gui_file_setup(bbci);
% 
% input:
%    bbci struct with fields
%    files     name of files as default (default: {})
%    player    the player number (default: 1);
%    setup     the name of a processing (default: [])
%    classDef  a classDef in the usual format {trg;className} (default: [])
% 
% output:
%    if button load is pressed:
%       the modified values of the input
%    if button exit is pressed:
%       files = -1, the other values are empty
%    if button cancel is pressed:
%       files = 0, the other values are empty
%    
% see also:
%    training_bbci_bet   
%
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 07/03/05
% $Id: gui_file_setup.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $


% SOME DEFAULTS
global gui_file_setup_status;
global EEG_RAW_DIR BBCI_DIR

if nargin<=0 | isempty(bbci)
  bbci = struct('train_file',{{}},'player',1,'setup',[],'classDef',{{}});
end

bbci = set_defaults(bbci,'train_file',{{}},'player',1,'setup',[],'classDef',{{}});

bbci_default = bbci;

% OPENING THE FIGURE
fig = figure;   
scs = get(0,'ScreenSize');
set(fig,'Position',[1 scs(4)/2 scs(3)-1 scs(4)/2-1]);
set(fig,'MenuBar','none');
set(fig,'NumberTitle','off');
set(fig,'Name','File-Setup');
set(fig,'Units','Normalized');
  
% THE FILE MENU  
fil = uicontrol('Units','Normalized','Position',[0.02,0.7,0.81,0.28]);
set(fil,'Style','listbox');
set(fil,'String',bbci.train_file);
set(fil,'FontUnits','Normalized');
set(fil,'FontSize',0.15);
set(fil,'Tooltipstring','files to load, add files by pressing button add on the right, delete files by pressing button delete on the right');
set(fil,'Max',1000);


% THE ADD BUTTON
add = uicontrol('Units','Normalized','Position',[0.85 0.85 0.12 0.13]);
set(add,'Style','pushbutton');
set(add,'String','add');
set(add,'FontUnits','Normalized');
set(add,'FontSize',0.6);
set(add,'Callback','global gui_file_setup_status;gui_file_setup_status = ''add'';');
set(add,'Tooltipstring','opens a file browser to add a file to the file list');

% THE DELETE BUTTON
del = uicontrol('Units','Normalized','Position',[0.85 0.7 0.12 0.13]);
set(del,'Style','pushbutton');
set(del,'String','delete');
set(del,'FontUnits','Normalized');
set(del,'FontSize',0.6);
set(del,'Callback','global gui_file_setup_status;gui_file_setup_status = ''del'';');
set(add,'Tooltipstring','delete marked files from the file list');

% THE RESET BUTTON
res = uicontrol('Units','Normalized','Position',[0.02 0.02 0.15 0.1]);
set(res,'Style','pushbutton');
set(res,'String','reset');
set(res,'FontUnits','Normalized');
set(res,'FontSize',0.6);
set(res,'Callback','global gui_file_setup_status;gui_file_setup_status = ''res'';');
set(res,'Tooltipstring','RESET all entries');


% THE CANCEL BUTTON
can = uicontrol('Units','Normalized','Position',[0.22 0.02 0.15 0.1]);
set(can,'Style','pushbutton');
set(can,'String','cancel');
set(can,'FontUnits','Normalized');
set(can,'FontSize',0.6);
set(can,'Callback','global gui_file_setup_status;gui_file_setup_status = ''can'';');
set(can,'Tooltipstring','Goes back to the last menu');

% SUBJECT BUTTON
sub = uicontrol('Units','Normalized','Position',[0.62 0.02 0.15 0.1]);
set(sub,'Style','pushbutton');
set(sub,'String','setup');
set(sub,'FontUnits','Normalized');
set(sub,'FontSize',0.6);
set(sub,'Callback','global gui_file_setup_status;gui_file_setup_status = ''sub'';');
set(sub,'Tooltipstring','Loads a subject specific setup file');

% LOAD BUTTON
loa = uicontrol('Units','Normalized','Position',[0.82 0.02 0.15 0.1]);
set(loa,'Style','pushbutton');
set(loa,'String','load');
set(loa,'FontUnits','Normalized');
set(loa,'FontSize',0.6);
set(loa,'Callback','global gui_file_setup_status;gui_file_setup_status = ''loa'';');
set(loa,'Tooltipstring','Loads this configuration')

% EXIT BUTTON
exi = uicontrol('Units','Normalized','Position',[0.42 0.02 0.15 0.1]);
set(exi,'Style','pushbutton');
set(exi,'String','exit');
set(exi,'FontUnits','Normalized');
set(exi,'FontSize',0.6);
set(exi,'Callback','global gui_file_setup_status;gui_file_setup_status = ''exi'';');
set(exi,'Tooltipstring','exits bbci_bet');



% GET ALL MARKERS AND VISUALIZE
[mrk,str] = getMarker(bbci.train_file);
mrklist = uicontrol('Units','Normalized','Position',[0.01,0.6,0.98,0.08]);
set(mrklist,'Style','text');
set(mrklist,'String',['Marker: ' str]);
set(mrklist,'HorizontalAlignment','left');
set(mrklist,'FontUnits','Normalized');
set(mrklist,'FontSize',0.7);
set(mrklist,'Tooltipstring','these markers are available in the files');

% PLAYER
pla = uicontrol('Units','Normalized','Position',[0.75,0.2,0.2,0.1]);
set(pla,'Style','popupmenu');
set(pla,'String',{'Player 1','Player 2'});
set(pla,'Value',bbci.player);
set(pla,'FontUnits','Normalized');
set(pla,'FontSize',0.7);
set(pla,'Tooltipstring','Choose player 1 or 2, player 2 has all channels with starting x, player 1 all others');

% PROCESSING
setu = uicontrol('Units','Normalized','Position',[0.75,0.33,0.2,0.1]);
set(setu,'Style','popupmenu');
set(setu,'Value',1);
set(setu,'Min',1);
set(setu,'Max',1);
d = dir([BBCI_DIR 'setups']);
ind = strmatch('setup',{d.name});
d = {d(ind).name};
for i = 1:length(d)
  d{i} = d{i}(7:end-2);
end
set(setu,'String',d);
if isempty(bbci.setup)
  bbci.setup = d{1};
  ind = 1;
else
  ind = find(strcmp(bbci.setup,d));
  if isempty(ind)
    bbci.setup = d{1};
    ind = 1;
  end
end
set(setu,'Value',ind);
set(setu,'FontUnits','Normalized');
set(setu,'FontSize',0.7);
set(setu,'Tooltipstring','Choose a processing');

% FEEDBACK
feed = uicontrol('Units','Normalized','Position',[0.75,0.46,0.20,0.1]);
set(feed,'Style','popupmenu');
set(feed,'Value',1);
set(feed,'Min',1);
set(feed,'Max',1);
d = dir([BBCI_DIR 'online/feedbacks/']);
ind = strmatch('bbci_bet_feedbacks',{d.name});
d = {d(ind).name};
for i = 1:length(d)
  d{i} = d{i}(20:end);
end
dd = d;
d = {};
for i = 1:length(dd);
  if strcmp(dd{i}(end-1:end),'.m')
    d = cat(2,d,{dd{i}(1:end-2)});
  end
end

set(feed,'String',d);
if isempty(bbci.feedback)
  bbci.feedback = d{1};
  ind = 1;
else
  ind = find(strcmp(bbci.feedback,d));
  if isempty(ind)
    bbci.feedback = d{1};
    ind = 1;
  end
end
set(feed,'Value',ind);
set(feed,'FontUnits','Normalized');
set(feed,'FontSize',0.7);
set(feed,'Tooltipstring','Choose a feedback');

% used markers
mr = uicontrol('Units','Normalized','Position',[0.01,0.5,0.08,0.08]);
set(mr,'Style','text');
set(mr,'String',['Marker: ' str]);
set(mr,'HorizontalAlignment','left');
set(mr,'FontUnits','Normalized');
set(mr,'FontSize',0.7);
co = get(mr,'BackgroundColor');
set(mr,'Tooltipstring','choose all markers you want. Type the number between (+/- 1 and 255) or a 8 byte string with *,0,1 with leading +/- to specify bytes. Between markers of different classes have a space, markers for the same classes are separated only by comma (no spaces!!!)');

% THE MARKER FIELD
mrkta = uicontrol('Units','Normalized','Position',[0.1,0.5,0.6,0.08]);
set(mrkta,'Style','edit');
set(mr,'Tooltipstring','choose all markers you want. Type the number between (+/- 1 and 255) or a 8 byte string with *,0,1 with leading +/- to specify bytes. Between markers of different classes have a space, markers for the same classes are separated only by comma (no spaces!!!)');

str = getMarkerString(bbci.classDef);


set(mrkta,'String',str);
set(mrkta,'HorizontalAlignment','left');
set(mrkta,'FontUnits','Normalized');
set(mrkta,'FontSize',0.7);
set(mrkta,'Callback','global gui_file_setup_status;gui_file_setup_status = ''mrk'';');

% VISUALIZE ALL MARKERS
cln = vis_classnames([],bbci.classDef);


% NOW WAIT FOR ENTRIES (gui_file_setup_status will change if something happens)
gui_file_setup_status = '';

while true;
  drawnow;
  pause(0.1);
  if ~isempty(gui_file_setup_status)
    switch gui_file_setup_status
     case 'add'
      % ADD A FILE TO THE FILE LIST
      fi = get(fil,'String');
      if length(fi)==0
        default_dir = EEG_RAW_DIR;
        d = dir(default_dir);
        d = {d.name};
        da = datevec(date);
        da(1) = da(1)-2000;
        da = sprintf('_%02d_%02d_%02d',da(1:3));
        ind = [];
        for i = 1:length(d)
          if ~isempty(strfind(d{i},da))
            ind = [ind,i];
          end
        end
        if length(ind)==1
          default_dir = [default_dir d{ind} '/'];
        end
      else
        fi = fi{end};
        default_dir = [fileparts(fi),'/'];
        if (isunix & default_dir(1)~='/') | (~isunix & default_dir(2)~=':')
          default_dir = [EEG_RAW_DIR default_dir];
        end
      end
      [filename,pathname] = uigetfile('*.eeg','Pick an eeg-file',default_dir);
      if ~isnumeric(filename)
        fi = get(fil,'String');
        filename = [pathname filename(1:end-4)];
        if strmatch(EEG_RAW_DIR,filename)
          filename = filename(length(EEG_RAW_DIR)+1:end);
        end
        ind = find(strcmp(filename,fi));
        if isempty(ind) 
          ind = length(fi)+1;
        end
        fi{ind} = filename;  
        set(fil,'String',fi);
        set(fil,'Value',ind);
        [mrk,str] = getMarker(get(fil,'String'));
        set(mrklist,'String',['Marker: ' str]);
      end
     case 'del'
      % DELETE FILES FROM FILELIST
      po = get(fil,'Value');
      if po>0
        str = get(fil,'String');
        str(po) = [];
        set(fil,'String',str);
        if po>length(str)
          set(fil,'Value',length(str));
        end
        [mrk,str] = getMarker(get(fil,'String'));
        set(mrklist,'String',['Marker: ' str]);
      end
     case 'can'
      % CANCEL -> files=0 and break
      bbci = 0;
      break;
     case 'loa'
      % LOAD THE DATA
      bbci.train_file = get(fil,'String');
      if ~iscell(bbci.train_file)
        bbci.train_file = {bbci.train_file};
      end
      bbci.player = get(pla,'Value');
      bbci.setup = get(setu,'Value');
      s = get(setu,'String');
      if ~iscell(s)
        s = s{1};
      end
      bbci.setup = s{bbci.setup};      
      cl = get(mrkta,'String');
      cl = parseMarkerString(cl);
      if iscell(cl) | cl~=-1
        bbci.classDef = cat(1,cl,cell(1,length(cl)));        
        for i = 1:size(cln,2)
          bbci.classDef{2,i} = get(cln(2,i),'String');
        end
        break;
      else
        set(mr,'BackgroundColor',[1 0 0]);        
      end
     case 'exi'
      % EXIT
      bbci = -1;
      break;
     case 'res'
      % RESET
      bbci = bbci_default;
      set(fil,'String',bbci.train_file);
      set(fil,'Value',min(1,length(bbci.train_file)));
      [mrk,str] = getMarker(get(fil,'String'));
      set(mrklist,'String',['Marker: ' str]);
      set(pla,'Value',bbci.player);
      d = dir([BBCI_DIR 'setups']);
      ind = strmatch('setup',{d.name});
      d = {d(ind).name};
      for i = 1:length(d)
        d{i} = d{i}(7:end-2);
      end
      if isempty(bbci.setup)
        ind = 1;
        bbci.setup = d{1};
      else
        ind = strcmp(bbci.setup,d);
        if isempty(ind)
          ind = 1;
          bbci.setup = d{1};
        end
      end
      set(setu,'Value',ind);
      str = getMarkerString(bbci.classDef);
      set(mrkta,'String',str);
      cln = vis_classnames(cln,bbci.classDef);      
     case 'sub'
      % LOAD A SUBJECT FILE
      [filename,pathname] = uigetfile('*.m','Pick an m-file',[BBCI_DIR 'subjects/']);
      if ~isnumeric(filename)
        aa = load_subject_file(filename,pathname);
        if ~isempty(aa)
          bbci = aa;
          jj1 = [];
          for jj = 1:length(bbci.train_file)
            if check_file(bbci.train_file{jj});
              jj1 = [jj1,jj];
            end
          end
          bbci.train_file = bbci.train_file(jj1);
          set(fil,'String',bbci.train_file);
          set(fil,'Value',min(1,length(bbci.train_file)));
          [mrk,str] = getMarker(get(fil,'String'));
          set(mrklist,'String',['Marker: ' str]);
          set(pla,'Value',bbci.player);
          d = dir([BBCI_DIR 'setups']);
          ind = strmatch('setup',{d.name});
          d = {d(ind).name};
          for i = 1:length(d)
            d{i} = d{i}(7:end-2);
          end
          if isempty(bbci.setup)
            ind = 1;
            bbci.setup = d{1};
          else
            ind = find(strcmp(bbci.setup,d));
            if isempty(ind)
              bbci.setup = d{1};
              ind = 1;
            end
          end
          set(setu,'Value',ind);
          str = getMarkerString(bbci.classDef);
          set(mrkta,'String',str);
          cln = vis_classnames(cln,bbci.classDef);                      
        end
      end
     case 'mrk'
      % CHANGE MARKER FIELDS
      set(mr,'BackgroundColor',co);
      cl = get(mrkta,'String');
      cl = parseMarkerString(cl);
      if iscell(cl) | cl~=-1
        %UPDATE CLASSDEF
        for i = 1:size(cln,2)
          bbci.classDef{2,i} = get(cln(2,i),'String');
        end
        nam = cell(1,length(cl));
        for i = 1:length(cl)
          ind = cellcmp(cl{i},bbci.classDef);
          if ~isempty(ind)
            nam{i} = bbci.classDef{2,ind};
          end
        end
        bbci.classDef = cat(1,cl,nam);
      else
        set(mr,'BackgroundColor',[1 0 0]);
      end
      set(mrkta,'String',getMarkerString(bbci.classDef));
      cln = vis_classnames(cln,bbci.classDef);
     otherwise
      error('wrong call');
    end
    
  end
  gui_file_setup_status = '';
end

% GOODBYE
close(fig);  
return;



function [mrks,str] = getMarker(files);
% THE FUNCTION getMarker reads out the files and present all markers
w = warning;
warning off;
if ~iscell(files)
  files = {};
end

toe = [];
for i = 1:length(files);
  mrk = readMarkerTable(files{i});
  toe = [toe,unique(mrk.toe)];
end
toe = unique(toe);

com = {};
for i = 1:length(files);
  mrk = readMarkerComments(files{i});
  com = [com,mrk.str];
end
com = unique(com);


mrks = {toe,com};

str = sprintf('%d, ',mrks{1});
str2 = sprintf('%s, ',mrks{2}{:});
str = [str,str2];
str = str(1:end-2);
warning(w);

return;



function str = getMarkerString(classes);
%GETMARKERSTRING performs a marker string

str = '';

if isempty(classes)
  return
end

classes = classes(1,:);

for i = 1:length(classes)
  if isnumeric(classes{i})
    classes{i} = num2cell(classes{i});
  end
  for j = 1:length(classes{i})
    if isnumeric(classes{i}{j})
      str = [str,int2str(classes{i}{j})];
    else
      str = [str,classes{i}{j}];
    end
    if j<length(classes{i})
      str = [str,','];
    end
  end
  if i<length(classes)
    str = [str ' '];
  end
end


return;


function cl = parseMarkerString(str);
% PARSE MARKER STRING

while length(str)>0 & str(1)==' ' 
  str = str(2:end);
end

c = [0,strfind(str,' '),length(str)+1];
cl = cell(1,length(c));
ind =[];
for i = 1:length(c)-1
  cl{i} = str(c(i)+1:c(i+1)-1);
  if ~isempty(cl{i})
    ind = [ind,i];
  end
end

cl = cl(ind);

for i = 1:length(cl)
  c = [0,strfind(cl{i},','),length(cl{i})+1];
  cc = cell(1,length(c));
  ind = [];
  for j = 1:length(c)-1
    cc{j} = cl{i}(c(j)+1:c(j+1)-1);
    if ~isempty(cc{j})
      ind = [ind,j];
    end
  end
  cc = cc(ind);
  cl{i} = cc;
end

for i = 1:length(cl)
  flag = true;
  for j = 1:length(cl{i})
    a = str2num(cl{i}{j});
    if ~isempty(a)
      if a>255 | a<-255 | a==0 | a~=round(a)
        cl = -1;
        return;
      end
      cl{i}{j} = a;
    else
      st = cl{i}{j};
      if st(1)=='-'  | st(1)=='+'
        st = st(2:end);
      end
      if length(st)~=8 | ~isempty(setdiff(unique(st),'*01'))
        cl = -1;
        return;
      end
      flag = false;
    end
  end
  if flag
    cl{i} = [cl{i}{:}];
  end
end

for i = 1:length(cl)
  cl{i} = unique(cl{i});
end

for i = 1:length(cl)
  for j = i+1:length(cl)
    if (isnumeric(cl{i}) & isnumeric(cl{j})) | (iscell(cl{i}) & iscell(cl{j}))
      c = intersect(cl{i},cl{j});
      if ~isempty(c)
        cl = -1;
        return;
      end
    end
  end
end

return;

function cln = vis_classnames(cln,classDef);
% VISUALIZE THE CLASSES
hei = 0.1;

for i = size(cln,2)+1:size(classDef,2)
  cln(1,i) = uicontrol('Units','Normalized','Position',[0.01+0.34*mod(i-1,2),0.45-ceil(0.5*i)*hei,0.1,hei-0.005]);
  set(cln(1,i),'Style','text');
  set(cln(1,i),'String',['class ' int2str(i) ':']);
  set(cln(1,i),'HorizontalAlignment','left');
  set(cln(1,i),'FontUnits','Normalized');
  set(cln(1,i),'FontSize',0.7); 
  cln(2,i) = uicontrol('Units','Normalized','Position',[0.12+0.34*mod(i-1,2),0.45-ceil(0.5*i)*hei,0.22,hei-0.005]);
  set(cln(2,i),'Style','edit');
  set(cln(2,i),'HorizontalAlignment','left');
  set(cln(2,i),'FontUnits','Normalized');
  set(cln(2,i),'FontSize',0.7); 
end

for i = size(classDef,2)+1:size(cln,2)
  delete(cln(:,i));
end

cln = cln(:,1:size(classDef,2));

hei = min(0.1,0.3/max(1,ceil(0.5*size(cln,2))));

for i = 1:size(cln,2)
  set(cln(1,i),'Position',[0.01+0.34*mod(i-1,2),0.45-ceil(0.5*i)*hei,0.1,hei-0.005]);
  set(cln(2,i),'Position',[0.12+0.34*mod(i-1,2),0.45-ceil(0.5*i)*hei,0.22,hei-0.005]);
  set(cln(2,i),'String',classDef{2,i});
  str = getMarkerString(classDef(:,i));
  set(cln(:,i),'Tooltipstring',['If desired the class with members ' str ' can be labelled by this name']);
end


return;

function ind = cellcmp(a,b);
% compares cell entries

ind = [];
for i = 1:length(b)
  if isnumeric(a) & isnumeric(b{1,i}) & length(a)==length(b{1,i})
    c = sort(a)-sort(b{1,i});
    c = sum(abs(c));
    if c==0
      ind = [ind,i];
    end
  elseif iscell(a) & iscell(b{1,i}) & length(a)==length(b{1,i})
    flag = true;
    for j = 1:length(a)
      if (isnumeric(a{j}) & isnumeric(b{1,i}{j}) & a{j}==b{1,i}{j}) | ...
         (~isnumeric(a{j}) & ~isnumeric(b{1,i}{j}) & strcmp(a{j},b{1,i}{j}))
        %nothing
      else
        flag = false;
      end
    end
    if flag
      ind = [ind,i];
    end
  end
end

            
return;

function flag = check_file(file);

global EEG_RAW_DIR 
if (isunix & file(1)~='/') | (~isunix & file(2)~=':')
  file = [EEG_RAW_DIR file];
end

file = [file '.eeg'];

flag = exist(file,'file');

