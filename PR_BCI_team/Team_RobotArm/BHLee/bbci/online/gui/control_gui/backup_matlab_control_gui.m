function backup_matlab_control_gui(setup);

global TMP_DIR
if isempty(TMP_DIR)
  if isunix
    TMP_DIR = '/tmp/';
  else
    TMP_DIR = 'c:\temp\';
  end
end

% tmp_file = sprintf('%smatlab_control_gui_backup_%04d_%02d_%02d_%02d_%02d_%02d.setup',TMP_DIR,round(datevec(now)));

if isfield(setup,'general')
  try,setup.general = rmfield(setup.general,'setup_list1');end
  try,setup.general = rmfield(setup.general,'setup_list1_default');end
  try,setup.general = rmfield(setup.general,'setup_list2');  end
  try,setup.general = rmfield(setup.general,'setup_list2_default');end
end

str = get_save_string(setup,'setup');
% fid = fopen(tmp_file,'w');
% fprintf(fid,'%s',str);
% fclose(fid);
