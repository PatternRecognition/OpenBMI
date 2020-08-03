function load_control_setup(fig,typ,vars,varargin)
% LOAD_CONTROL_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% usage:
%    load_control_setup(fig,typ,vars);
%
% IN: 
%    fig   - figure handle of the matlab_control_gui 
%    typ   - string specifying the way in which current values of 
%            the gui fields are replaced/extended by the setup file.
%            possible entries:
%            'default': completely ignore the current setup.Use the
%                       entries of the setup file instead.
%            'merge':   take the fields from the new setup. If these
%                       fields exist in the current setup, use the values.
%            'add':     only add fields from the setup file. Don't 
%                       remove fields from the current setup.
%    vars  - names of the fields of the setup which should be changed.
%            cell array which may have values 
%            'all' or several of the values 'control_player1',
%            'control_player2','graphic_player1','graphic_player2',
%            'general'.
%    varargin - optional setup_file directly given as input argument
%    instead of manually loading it. It should be the path of the file
%    relative to [BCI_DIR online/setups/'];
%
%

% kraulem 07/05
% $Id: load_control_setup.m,v 1.4 2007/04/23 11:12:14 neuro_cvs Exp $

global BBCI_DIR

opt = propertylist2struct(varargin{:});

setup = control_gui_queue(fig,'get_setup');
setup = set_defaults(setup,'general',struct);
setup.general = set_defaults(setup.general,'setup_directory',[BBCI_DIR 'setups/']);
control_gui_queue(fig,'set_setup',setup);

if ~iscell(vars)
  vars = {vars};
end
if length(vars)==1
  % This should be the usual case.
  switch(vars{1})
   case 'all'
    % only .setup files should be loaded.
    suffix = '.setup';
    vars = {'control_player1','control_player2','graphic_player1',...
	    'graphic_player2','graphic_master','general'};
    player = 0;
   case {'control_player1','control_player2'}
    % only .csetup files
    suffix = '.csetup';
    player = str2num(vars{1}(end));
   case {'graphic_player1','graphic_player2'}
    % only .gsetup files.
    suffix = '.gsetup';
    player = str2num(vars{1}(end));
  case 'graphic_master'
    suffix = '.msetup';
    player = [];
  end
end


if isfield(opt,'setup_file')
  if isabsolutepath(opt.setup_file);
    file = opt.setup_file;
  else
    file = [setup.general.setup_directory opt.setup_file];
  end
  setup.setup_directory = file(1:max(findstr(file,'/')));
else
    % user interaction:
    set(fig,'Visible','off');

    % this line has been changed by claudia. Before was:
    % [file,pa] = uigetfile(['*' suffix],'Specify the name of a setup file',...
    % 		      setup.general.setup_directory);
    % but we want to load setup files always from "BCI_DIR/bbci_bet/setups"
    % directory, so:
    [file,pa] = uigetfile(['*' suffix],'Specify the name of a setup file',...
        [setup.general.setup_directory]);
    set(fig,'Visible','on');
    file = [pa,file];
    setup.setup_directory = pa;
end


if ~ischar(file)
  % user pressed cancel. Don't do anything.
  return
elseif ischar(file)
  if length(file)<=6 | ~strcmp(file(end-length(suffix)+1:end),suffix)
    file = [file,suffix];
  end
end

if ~exist(file,'file')
  % file does not exist. Do something, but don't try to load it.
  bbci_bet_message('File %s does not exist.',file);
  return
end


control_gui_queue(fig,'set_setup',setup);
setup_old = setup;

% now load the stored setup:
clear setup;
fid = fopen(file,'r');
F = fread(fid);
eval(char(F'));

setup_new = setup;

if ~isempty(player)
  switch player
    case {1,2}
      % only one field is available.
      % rename this field to contain the player number.
      setup_new = setfields(setup_new,vars{1},getfields(setup_new,vars{1}(1:end-1)));
      setup_new = rmfield(setup_new,vars{1}(1:end-1));
  end
end

setup_new.control_player1.machine = setup_old.control_player1.machine;
setup_new.control_player1.port = setup_old.control_player1.port;
if isfield(setup_old,'control_player2');
  setup_new.control_player2.machine = setup_old.control_player2.machine;
  setup_new.control_player2.port = setup_old.control_player2.port;
end

setup_new.graphic_player1.machine = setup_old.graphic_player1.machine;
setup_new.graphic_player1.port = setup_old.graphic_player1.port;
setup_new.graphic_player1.fb_port = setup_old.graphic_player1.fb_port;

if isfield(setup_old.graphic_player1,'directions')
    % copy directions, so just one setup file is needed Claudia
    setup_new.graphic_player1.feedback_opt.directions = setup_old.graphic_player1.feedback_opt.directions;
end

if isfield(setup_old,'graphic_player2');
  setup_new.graphic_player2.machine = setup_old.graphic_player2.machine;
  setup_new.graphic_player2.port = setup_old.graphic_player2.port;
  setup_new.graphic_player2.fb_port = setup_old.graphic_player2.fb_port;
  if isfield(setup_old.graphic_player2,'feedback_opt');
      if isfield(setup_old.graphic_player2,'directions') 
          % copy directions, so just one setup file is needed Claudia
          setup_new.graphic_player2.feedback_opt.directions = setup_old.graphic_player2.feedback_opt.directions;
      end
  end
end

try,setup_new.general.setup_list1 = setup_old.general.setup_list1;end
try,setup_new.general.setup_list2 = setup_old.general.setup_list2;end
try,setup_new.general.setup_list1_default = setup_old.general.setup_list1_default;end
try,setup_new.general.setup_list2_default = setup_old.general.setup_list2_default;end
try,setup_new.general.active1 = setup_old.general.active1;end
try,setup_new.general.active2 = setup_old.general.active2;end



setup = setup_old;
for ii = 1:length(vars)
  % find out which fields overlap
  if ~strcmp(vars{ii},'general')
    fi_new = getfields(getfields(setup_new,vars{ii}),'fields');
    fi_old = getfields(getfields(setup,vars{ii}),'fields');
  else
    fi_new = fieldnames(getfields(setup_new,vars{ii}));
    fi_old = fieldnames(getfields(setup,vars{ii}));
  end
  fi_int = intersect(fi_new,fi_old);
  fi_rm = setdiff(fi_old,fi_new);
  fi_add = setdiff(fi_new,fi_old);
  switch typ
   case 'default'
    % take every value of the setup file. Ignore the current setup.
    % that means setup_new doesn't need changes.
    setup = setfields(setup,vars{ii},getfields(setup_new,vars{ii}));
   case 'merge'
    % keep the current field values. Remove 
    % the current fields of the setup which are not in the stored setup.
    if ~strcmp(vars{ii},'general')
      fields = {};
      fields_help_new = getfields(getfields(setup_new,vars{ii}),'fields_help');
      fields_help_old = getfields(getfields(setup_old,vars{ii}),'fields_help');
      fields_help = {};
    end
    for jj = 1:length(fi_add)
      setup = setfields(setup,vars{ii},setfields(getfields(setup,vars{ii}), fi_add{jj},getfields(getfields(setup_new,vars{ii}),fi_add{jj})));
      if ~strcmp(vars{ii},'general')
	% now update fields and fields_help.
	fields = {fields{:},fi_add{jj}};
	fields_help_ind = find(strcmp(getfields(getfields(setup_new,vars{ii}),'fields'),fields{end}));
	fields_help = {fields_help{:},fields_help_new{fields_help_ind}};
      end
    end
    if ~strcmp(vars{ii},'general')
      for jj = 1:length(fi_int)
	fields = {fields{:},fi_int{jj}};
	fields_help_ind = find(strcmp(getfields(getfields(setup_old,vars{ii}),'fields'),fields{end}));
	fields_help = {fields_help{:},fields_help_old{fields_help_ind}};
      end
      % remove current fields.
      setup = setfields(setup,vars{ii},rmfield(getfields(setup,vars{ii}),fi_rm));
      % set fields and fields_help.
      setup = setfields(setup,vars{ii},setfields(getfields(setup,vars{ii}),'fields',fields));
      setup = setfields(setup,vars{ii},setfields(getfields(setup,vars{ii}),'fields_help',fields_help));
    end
   case 'add' 
    % replace the current field values with the stored values. Keep 
    % the current fields of the setup.
    if ~strcmp(vars{ii},'general')
      fields = getfields(getfields(setup_old,vars{ii}),'fields');
      fields_help_new = getfields(getfields(setup_new,vars{ii}),'fields_help');
      fields_help_old = getfields(getfields(setup_old,vars{ii}),'fields_help');
      fields_help = fields_help_old;
    end
    for jj = 1:length(fi_add)
      setup = setfields(setup,vars{ii},setfields(getfields(setup,vars{ii}), fi_add{jj},getfields(getfields(setup_new,vars{ii}),fi_add{jj})));
      if ~strcmp(vars{ii},'general')
	% now update fields and fields_help.
	fields = {fields{:},fi_add{jj}};
	fields_help_ind = find(strcmp(getfields(getfields(setup_new,vars{ii}),'fields'),fields{end}));
	fields_help = {fields_help{:},fields_help_new{fields_help_ind}};
      end
    end
    if ~strcmp(vars{ii},'general')
      % set fields and fields_help.
      setup = setfields(setup,vars{ii},setfields(getfields(setup,vars{ii}),'fields',fields));
      setup = setfields(setup,vars{ii},setfields(getfields(setup,vars{ii}),'fields_help',fields_help));
    end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % transmit the new setup to the gui.

  control_gui_queue(fig,'set_setup',setup);
  % redraw everything.
  gui_part = vars{ii};
  switch(gui_part)
   case 'general'
    plot_general_gui(fig);
   case {'control_player1','control_player2'}
    plot_control_gui(fig,str2num(gui_part(end)));
   case {'graphic_player1','graphic_player2'}
    plot_graphic_gui(fig,str2num(gui_part(end)));
  case {'graphic_master'}
    plot_master_gui(fig);
  end
  setup = control_gui_queue(fig,'get_setup');
end
% get back to the first one of the windows:
activate_control_gui(fig,vars{end});

return
