function save_control_setup(fig,vars);
% SAVE_CONTROL_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% generates a file input dialogue and saves a setup file to the
% intended destination file.
%
% usage: 
%      save_control_setup(fig,vars)
%
% input:
%      fig   - array of figure handles. 
%      vars  - cell array containing strings from
%               'all', 'control_player1','control_player2','graphic_player1'
%               'graphic_player2'.
%
% Matthias Krauledat
% $Id: save_control_setup.m,v 1.2 2006/05/05 12:44:05 neuro_cvs Exp $

global BBCI_DIR
% query the gui for the current setup.
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
	    'graphic_player2','general','graphic_master'};
    setup_save = setup;
   case {'control_player1','control_player2'}
    % only .csetup files
    suffix = '.csetup';
    fi_rm = setdiff(fieldnames(setup),vars);
    setup_save = rmfield(setup,fi_rm);
    setup_save = setfield(setup_save,vars{1}(1:end-1),getfield(setup_save,vars{1}));
    setup_save = rmfield(setup_save,vars{1});
   case {'graphic_player1','graphic_player2'}
    % only .gsetup files.
    suffix = '.gsetup';
    fi_rm = setdiff(fieldnames(setup),vars);
    setup_save = rmfield(setup,fi_rm);
    setup_save = setfield(setup_save,vars{1}(1:end-1),getfield(setup_save,vars{1}));   
    setup_save = rmfield(setup_save,vars{1});
  case 'graphic_master'
    suffix = '.msetup';
    fi_rm = setdiff(fieldnames(setup),vars);
    setup_save = rmfield(setup,fi_rm);
    

  end
end
% get user input
set(fig,'Visible','off');
% this line has been changed by claudia. Before was:
% [file,pa] = uigetfile(['*' suffix],'Specify the name of a setup file', setup.general.setup_directory);
% but we want to load setup files always from "BCI_DIR/bbci_bet/setups"
% directory, so:
[file,pa] = uiputfile(['*' suffix],'Specify the name of a setup file', setup.general.setup_directory);

set(fig,'Visible','on');

if ~ischar(file)
  % user pressed cancel. Don't do anything.
  return
elseif ischar(file)
  if length(file)<=6 | ~strcmp(file(end-length(suffix)+1:end),suffix)
    file = [file,suffix];
  end
end
setup.setup_directory = pa;
control_gui_queue(fig,'set_setup',setup);

% save the setup into a file.
if isfield(setup_save,'general')
  try,setup_save.general = rmfield(setup_save.general,'setup_list1');end
  try,setup_save.general = rmfield(setup_save.general,'setup_list2');  end
  try,setup_save.general = rmfield(setup_save.general,'setup_list1_default');end
  try,setup_save.general = rmfield(setup_save.general,'setup_list2_default');end
  try,setup_save.general = rmfield(setup_save.general,'active1');end
  try,setup_save.general = rmfield(setup_save.general,'active2');end
  
end

str = get_save_string(setup_save,'setup');
fid = fopen([pa,file],'w');
fprintf(fid,'%s',str);
fclose(fid);


set(fig,'Visible','on');
return



