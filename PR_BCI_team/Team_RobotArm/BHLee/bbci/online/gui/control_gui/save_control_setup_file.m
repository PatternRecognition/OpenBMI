function save_control_setup_file(fig,string);
% SAVEC_CONTROL_SETUP_FILE ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% usage:
%    save_control_setup_file(fig,string);
% 
% input:
%    fig    - handle to the figure containing the matlab gui.
%    string - a string containing the path to a file under which
%             name the setup should be stored.
% 
%

% Matthias Krauledat
% $Id: save_control_setup_file.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

%activate_all_entries(fig);
%setup = control_gui_queue(fig,'get_setup');

if isfield(setup,'general')
  setup.general = rmfield(setup.general,'setup_list1');
  setup.general = rmfield(setup.general,'setup_list1_default');
  setup.general = rmfield(setup.general,'setup_list2');  
  setup.general = rmfield(setup.general,'setup_list2_default')
end

str = get_save_string(setup,'setup');
fid = fopen(string,'w');
fprintf(fid,'%s',str);
fclose(fid);

return;
