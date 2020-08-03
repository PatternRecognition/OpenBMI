function vp_number= acq_vpcounter(session_name, cmd)
%ACQ_VPCOUNTER - Implement counter for participants within a study
%
%Synopsis:
%  VP_NUMBER= acq_vpcounter(SESSION_NAME, 'new_vp');
%  acq_vpcounter(SESSION_NAME, 'close');
%  acq_vpcounter(SESSION_NAME, 'reset');
%
%Remark:
%  If VP_CODE (global variable) is 'temp' (ignoring case) or 'VPtemp',
%  VP_NUMBER is returned as 1, and the VP_COUNTER file is not modified.


global DATA_DIR VP_CODE

session_name= strrep(session_name, '/', '-');
session_name= strrep(session_name, '\', '-');
session_name= strrep(session_name, '--', '-');

vp_counter_path= fullfile(DATA_DIR, 'vp_counter_files');
if ~exist(vp_counter_path, 'dir'),
  mkdir(vp_counter_path);
end

vp_counter_file= fullfile(vp_counter_path, [session_name '_VP_Counter']);

if strcmp(cmd, 'reset'),
  delete([vp_counter_file '.mat']);
  return;
end

if strcmpi(VP_CODE, 'temp') || strcmpi(VP_CODE, 'vptemp'),
  if strcmp(cmd, 'close'),
    fprintf('VP-TEMP: counter file not modified.\n');
  else
    vp_number= 1;
    fprintf('VP-TEMP: using counter value 1.\n');
  end
  return;
else
  if exist([vp_counter_file '.mat']),
    load(vp_counter_file, 'vp_number');
  else
    vp_number= 0;
  end
  vp_number= vp_number + 1;
end

switch(cmd),
 case 'close',
  save(vp_counter_file, 'vp_number');
  fprintf('Counter file for VP #%d saved.\n', vp_number);
 case 'new_vp',
  fprintf('New VP #%d.\n', vp_number);
 otherwise,
  error('unknown command ''%s''.', cmd);
end
