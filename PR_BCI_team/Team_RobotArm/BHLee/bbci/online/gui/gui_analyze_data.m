function [bbci,active] = gui_analyze_data(bbci,active,feature_calc);
%GUI_ANALYZE_DATA starts a gui for setting up some general variables 
%within training_bbci_bet to prepare analysis
%
% NOTE: DO NOT USE THIS FUNCTION ON ITS OWN!!! USE IT IN TRAINING_BBCI_BET
%
% usage: 
% [vars,classes,active,withgraphics,withclassification] = gui_analyze_data(used_vars,vars,varias,classNames,feature_calc,classes,active,withgraphics,withclassification,statement,nclassesrange);
%
% input:
%  used_vars    name of variables to use, e.g. {'csp'}
%  vars         the values of the used_vars as cell
%  varias       the variable in used_vars which are variable
%  classNames   the available classNames
%  feature_calc are feature calculated so far???
%  classes      default classes
%  active       active figures
%  withgraphics boolean, if graphic analysis is on
%  withclassification boolean, if classification analysis is on
%  statement    last statement about classification
%  nclassesrange range number of classes should be in
%
% output:
%  vars         the modified vars
%  classes      the chosen classes
%  active       the remained active figures
%  withgraphics the withgraphics flag
%  withclassification the withclassification flag
%
% see also : training_bbci_bet
%
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 08/03/2005
% $Id: gui_analyze_data.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $


%some defaults

global gui_analyze_data_status BBCI_DIR;

bbci_default = bbci;

% build the gui
fig = figure(100);
close(fig);
fig = figure(100);
clf;
scs = get(0,'ScreenSize');
set(fig,'MenuBar','none');
set(fig,'NumberTitle','off');
set(fig,'Name','Analysis-Setup');
set(fig,'Position',[(scs(3)-50)*0.5 (scs(4)-50)*0.5 scs(3)-50 scs(4)-50]);
set(fig,'Units','Normalized');

%PREPARE BUTTON
prep = uicontrol('Units','Normalized','Position',[0.01 0.01 0.18 0.08]);
set(prep,'Style','pushbutton');
set(prep,'String','Prepare');
set(prep,'FontUnits','Normalized');
set(prep,'FontSize',0.6);
set(prep,'Callback','global gui_analyze_data_status;gui_analyze_data_status = ''prep'';');
set(prep,'Tooltipstring','Starts Prepare Menu for changing the dataset');


%EXIT BUTTON
exi = uicontrol('Units','Normalized','Position',[0.21 0.01 0.18 0.08]);
set(exi,'Style','pushbutton');
set(exi,'String','Exit');
set(exi,'FontUnits','Normalized');
set(exi,'FontSize',0.6);
set(exi,'Callback','global gui_analyze_data_status;gui_analyze_data_status = ''exit'';');
set(exi,'Tooltipstring','Exits training_bbci_bet');

%RESET BUTTON
res = uicontrol('Units','Normalized','Position',[0.41 0.01 0.18 0.08]);
set(res,'Style','pushbutton');
set(res,'String','Reset');
set(res,'FontUnits','Normalized');
set(res,'FontSize',0.6);
set(res,'Callback','global gui_analyze_data_status;gui_analyze_data_status = ''reset'';');
set(res,'Tooltipstring','Reset all values');

%FINISH BUTTON
fin = uicontrol('Units','Normalized','Position',[0.61 0.01 0.18 0.08]);
set(fin,'Style','pushbutton');
set(fin,'String','Finish');
set(fin,'FontUnits','Normalized');
set(fin,'FontSize',0.6);
set(fin,'Callback','global gui_analyze_data_status;gui_analyze_data_status = ''finish'';');
set(fin,'Tooltipstring','Prepare a file for apply_bbci_bet within this setup');

%Analyze BUTTON
ana = uicontrol('Units','Normalized','Position',[0.81 0.01 0.18 0.08]);
set(ana,'Style','pushbutton');
set(ana,'String','Analyze');
set(ana,'FontUnits','Normalized');
set(ana,'FontSize',0.6);
set(ana,'Callback','global gui_analyze_data_status;gui_analyze_data_status = ''analyze'';');
set(ana,'Tooltipstring','Analyze this setup');

% VARIABLE_FIELDS_BUTTONS
wid = 0.75;
fi = bbci.setup_opts.editable(:,1);
hei = min(0.1,0.9/(length(fi)+1));
h = uicontrol('units','normalized','position',[0.1,0.99-hei,wid*0.49,hei*0.9]);
set(h,'Style','text');
set(h,'String',bbci.setup);
set(h,'FontUnits','normalized');
set(h,'FontSize',0.5);

fields = zeros(1,length(fi));
for j = 1:length(fi)
  h = uicontrol('units','normalized','position',[0.1,0.99-(j+1)*hei,wid*0.49,hei*0.9]);
  set(h,'Style','text');
  set(h,'String',bbci.setup_opts.editable{j,1});
  set(h,'Tooltipstring',bbci.setup_opts.editable{j,2});
  set(h,'FontUnits','normalized');
  set(h,'FontSize',0.3);
  fields(j) = uicontrol('units','normalized','position',[0.1+0.5*wid,0.99-(j+1)*hei,wid*0.49,hei*0.9]);
  set(fields(j),'Style','edit');
  set(fields(j),'Tooltipstring',bbci.setup_opts.editable{j,2});
  set(fields(j),'FontUnits','normalized');
  set(fields(j),'FontSize',0.3);
    set(fields(j),'Callback','global gui_analyze_data_status;gui_analyze_data_status = ''update'';');
end

set_values(fields,bbci.setup_opts);

% CLASS BUTTONS
cla = uicontrol('units','normalized','position',[0.85,0.75,0.145,0.23]);
set(cla,'Style','listbox');
set(cla,'FontUnits','normalized');
set(cla,'FontSize',0.1);
set(cla,'String',bbci.classDef(2,:));
set(cla,'Tooltipstring','choose the classes');
set(cla,'Min',0);
set(cla,'Max',length(bbci.classDef(2,:)));
set(cla,'Callback','global gui_analyze_data_status;gui_analyze_data_status = ''update'';');
activate_classes(cla,bbci.classDef(2,:),bbci.classes);
if length(bbci.classes)<bbci.nclassesrange(1) | length(bbci.classes)>bbci.nclassesrange(2)
  set(cla,'BackgroundColor',[1 0 0]);
  set(ana,'Enable','off');
end


% WITHGR and WITHCL
gr = uicontrol('units','normalized','position',[0.005,0.2,0.09,0.08]);
set(gr,'Style','checkbox');
set(gr,'FontUnits','normalized');
set(gr,'FontSize',0.2);
set(gr,'String','Graphics');
set(gr,'Tooltipstring','Analysis with graphics???');
set(gr,'Value',bbci.withgraphics);
wcl = uicontrol('units','normalized','position',[0.005,0.1,0.09,0.08]);
set(wcl,'Style','checkbox');
set(wcl,'FontUnits','normalized');
set(wcl,'FontSize',0.2);
set(wcl,'String','Classification');
set(wcl,'Tooltipstring','Analysis with classification???');
set(wcl,'Value',bbci.withclassification);

% IMPORT BUTTON
imp = uicontrol('units','normalized','position',[0.85,0.3,0.145,0.08]);
set(imp,'Style','pushbutton');
set(imp,'FontUnits','normalized');
set(imp,'FontSize',0.3);
set(imp,'String','Import');
set(imp,'Tooltipstring','Import a setup');
set(imp,'Callback','global gui_analyze_data_status; gui_analyze_data_status=''import'';');

% EXPORT BUTTON
exp = uicontrol('units','normalized','position',[0.85,0.2,0.145,0.08]);
cob = get(exp,'BackgroundColor');
set(exp,'Style','pushbutton');
set(exp,'FontUnits','normalized');
set(exp,'FontSize',0.3);
set(exp,'String','Export');
set(exp,'Tooltipstring','Export a setup');
set(exp,'Callback','global gui_analyze_data_status; gui_analyze_data_status=''export'';');


% WINDOW BUTTON
fignames = handlefigures('get');
for i = 1:length(fignames)
  handlefigures('changes',fignames{i},'CloseRequestFcn',sprintf('handlefigures(''vis'',''off'',''%s'');global gui_analyze_data_status; gui_analyze_data_status=-%i;',fignames{i},i));
end
hei = min(0.05,0.6/max(1,length(fignames)));
figis = zeros(1,length(fignames));
if length(fignames)>length(active)
  active = cat(2,active,ones(length(fignames)-length(active)));
end
for i = 1:length(fignames)
  figis(i) = uicontrol('units','normalized','position',[0.01,1-hei*i,0.08,hei]);
  set(figis(i),'FontUnits','normalized');
  set(figis(i),'FontSize',0.3);
  set(figis(i),'Style','checkbox');
  set(figis(i),'String',fignames{i});
  set(figis(i),'Value',active(i));
  if active(i)
    handlefigures('vis','on',fignames{i});
  else
    handlefigures('vis','off',fignames{i});
  end
  set(figis(i),'Tooltipstring','activate/deactivate the figure');
  set(figis(i),'Callback',['global gui_analyze_data_status; gui_analyze_data_status=' int2str(i) ';']);
end


%Figure button
set(fig,'Visible','off');
drawnow;
set(fig,'Visible','on');

figcl = uicontrol('units','normalized','position',[0.005,0.305,0.09,0.04]);
set(figcl,'FontUnits','normalized');
set(figcl,'FontSize',0.3);
set(figcl,'Style','pushbutton');
set(figcl,'String','close all');
set(figcl,'Tooltipstring','deactivate all figures');
set(figcl,'Callback',['global gui_analyze_data_status; gui_analyze_data_status=''closefig'';']);

figop = uicontrol('units','normalized','position',[0.005,0.355,0.09,0.04]);
set(figop,'FontUnits','normalized');
set(figop,'FontSize',0.3);
set(figop,'Style','pushbutton');
set(figop,'String','open all');
set(figop,'Tooltipstring','activate all figures');
set(figop,'Callback',['global gui_analyze_data_status; gui_analyze_data_status=''openfig'';']);
if length(fignames)==0
  set(figcl,'Visible','off','Enable','off');
  set(figop,'Visible','off','Enable','off');
end

%STATEMEMT MESSAGE
rem = uicontrol('units','normalized','position',[0.85,0.4,0.145,0.3]);
set(rem,'FontUnits','normalized');
set(rem,'FontSize',0.05);
set(rem,'Style','text');
set(rem,'String',sprintf('%s\n',bbci.analyze.message));
set(rem,'Tooltipstring','last results');



active = [];
classes = {};

gui_analyze_data_status = [];

updated = true;

% GO FOR THE LOOP
while true;
  if updated
    % CHECK INPUTS; SINCE THEY WERE UPDATED
    flag = true;
    
    bbci_o = bbci;
    for j = 1:size(bbci.setup_opts.editable,1);
      try
        eval(sprintf('bbci_o.setup_opts.%s = %s;',bbci.setup_opts.editable{j,1},get(fields(j),'String')));
        set(fields(j),'BackgroundColor',cob);
      catch
        set(fields(j),'BackgroundColor',[1 0 0]);
        flag = false;
      end
    end
    if flag
      bbci = bbci_o;
    end
    bbci.classes = bbci.classDef(2,(get(cla,'Value')));
    if length(bbci.classes)<bbci.nclassesrange(1) | length(bbci.classes)>bbci.nclassesrange(2)
      set(cla,'Backgroundcolor',[1 0 0]);
      flag = false;
    else
      set(cla,'Backgroundcolor',cob);
    end
    cm = obj_cmp(bbci,bbci_default);
    if flag*cm*feature_calc
      set(fin,'Enable','on');
    else
      set(fin,'Enable','off');
    end
    if flag*(~feature_calc+~cm)
      set(ana,'Enable','on');
    else
      set(ana,'Enable','off');
    end
  end
  updated = false;
  pause(0.1);
  if ~isempty(gui_analyze_data_status)
    if isnumeric(gui_analyze_data_status)
      if gui_analyze_data_status<0
        set(figis(-gui_analyze_data_status),'Value',0);
      else
        % FIGURES ACTIVATION WERE CHANGED
        active(gui_analyze_data_status) = get(figis(gui_analyze_data_status),'Value');
        if active(gui_analyze_data_status)
          handlefigures('vis','on',fignames{gui_analyze_data_status});
        else
          handlefigures('vis','off',fignames{gui_analyze_data_status});
        end
      end
    else
      switch gui_analyze_data_status
       case 'closefig'
        % CLOSE ALL FIGURES
        handlefigures('vis','off');
        for i = 1:length(figis)
          set(figis(i),'Value',0);
        end
       case 'openfig'
        % OPEN ALL FIGURES
        handlefigures('vis','on');
        for i = 1:length(figis)
          set(figis(i),'Value',1);
        end
        set(fig,'Visible','off');
        drawnow;
        set(fig,'Visible','on');
       case 'update'
        % one value was updated
        updated = true;
       case 'prep'
        % reload data
        bbci = 'prepare';
        close(fig);
        return;
       case 'exit'
        % GOODBYE
        bbci = 'exit';
        close(fig);
        return;
       case 'reset'
        % RESET ALL VALUES
        bbci = bbci_default;
        set_values(fields,bbci.setup_opts);
        activate_classes(cla,bbci.classDef(2,:),bbci.classes);
        if length(bbci.classes)<bbci.nclassesrange(1) | length(bbci.classes)>bbci.nclassesrange(2)
          set(cla,'Backgroundcolor',[1 0 0]);
          flag = false;
        else
          set(cla,'Backgroundcolor',cob);
          flag = true;
        end
        if feature_calc*flag
          set(fin,'Enable','on');
        end
        if flag
          set(ana,'Enable','on');
        else
          set(ana,'Enable','off');
        end
       case 'finish'
        % FINISH THIS SETUP
        flag = true;
        bbci_o= bbci;
        for j = 1:size(bbci.setup_opts.editable,1);
          try
            eval(sprintf('bbci_o.setup_opts.%s = %s;',bbci.setup_opts.editable{j,1},get(fields(j),'String')));
            set(fields(j),'BackgroundColor',cob);
          catch
            set(fields(j),'BackgroundColor',[1 0 0]);
            flag = false;
          end
        end
        if flag
          bbci = bbci_o;
        end
        bbci.classes = bbci.classDef(2,get(cla,'Value'));
        if length(bbci.classes)<bbci.nclassesrange(1) | length(bbci.classes)>bbci.nclassesrange(2)
          set(cla,'Backgroundcolor',[1 0 0]);
          flag = false;
        else
          set(cla,'Backgroundcolor',cob);
        end
        if flag*obj_cmp(bbci,bbci_default)*feature_calc;
          bbci = 'finish';
          close(fig);
          return;
        end
       case 'analyze'
        % ANALYZE THIS SETUP (GET DATA FIRST)
        flag = true;
        bbci.classes = bbci.classDef(2,get(cla,'Value'));
        if length(bbci.classes)<bbci.nclassesrange(1) | length(bbci.classes)>bbci.nclassesrange(2)
          flag = false;
        end
        for i = 1:length(figis)
          active(i) = get(figis(i),'Value');
        end
        bbci_o = bbci;
        for j = 1:size(bbci.setup_opts.editable,1);
          try
            eval(sprintf('bbci_o.setup_opts.%s = %s;',bbci.setup_opts.editable{j,1},get(fields(j),'String')));
            set(fields(j),'BackgroundColor',cob);
          catch
            set(fields(j),'BackgroundColor',[1 0 0]);
            flag = false;
          end
        end
        if flag*(~feature_calc+~obj_cmp(bbci,bbci_default))
          bbci = bbci_o;
          bbci.withgraphics = get(gr,'Value');
          bbci.withclassification = get(wcl,'Value');
          close(fig);
          return;
        else
          set(ana,'Enable','off');
        end
       case 'import'
        % IMPORT A FILE
        [fi,pa] = uigetfile('.mat','Pick a file',[BBCI_DIR 'subjects/']);
        if ~isnumeric(fi)
          S = load([pa,fi]);
          ind = strcmp(S.bbci.setup,bbci.setup);
          if ~isempty(ind)
            fi = fieldnames(S.setup_opts);
            for j = 1:length(fi)
              eval(sprintf('bbci.setup_opts.%s = S.bbci.setup_opts.%s;',ind,fi{j},fi{j}));
            end
          end
          set_values(fields,bbci.setup_opts);
          updated = true;
        end
       case 'export'
        %EXPORT A FILE
        [fi,pa] = uiputfile({'.mat'},'Select a file',[BBCI_DIR 'subjects/']);
        if ~isnumeric(fi)
          save([pa,fi],'bbci','vars');
        end
       otherwise
        error('not implemented so far');
      end
    end
    gui_analyze_data_status = [];
  end
end

% THIS PROGRAM SHOULD NEVER REACH THIS POINT!!! BUT IT LOOKS BETTER
return;



function set_values(fie,setup_opts)
%set values in the fields
for j = 1:length(fie)
  eval(sprintf('w=setup_opts.%s;',setup_opts.editable{j}));
  set(fie(j),'String',toString(w));
end

return;

function activate_classes(cla,nam,cl);
%activate some classes
ind = [];
for i = 1:length(cl)
  ind = [ind,find(strcmp(cl{i},nam))];
end

set(cla,'Value',ind);

return;

