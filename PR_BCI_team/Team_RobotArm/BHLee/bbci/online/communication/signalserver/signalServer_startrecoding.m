function  out= signalServer_startrecoding(fname, varargin)
% SIGNALSERVER_STARTRECODING 
% Starts a recording with the TOBI signalServer
% The SignalServer has to be running (start_signalserver('my_confic.xml') )
% to be done: extract fs from signalserver-settings
% INPUT:
%    fname: filename (folder will be equal to TODAY_DIR !!)
% OUTPUT:
%    actually chosen filename
% 
% Johannes 03/2011

% Benjamin 17|11|2011: Get fs from signal server


global VP_CODE TODAY_DIR

%check filename already exists, if so, change fname to fnameXX
if ~exist([TODAY_DIR fname '.vhdr'], 'file'),
    final_fname = fname;
else
    fileId = 2;
    while exist(sprintf('%s%s%02i.vhdr', TODAY_DIR, fname, fileId), 'file'),
        fileId = fileId + 1;
    end
    final_fname = sprintf('%s%02i', fname, fileId);
end
out= [TODAY_DIR final_fname];
disp(['EEG data will be saved in ' out])

[sig_info, dmy, dmy]= mexSSClient('localhost',9000,'tcp');
mexSSClient('close');
fs= sig_info(1);
cmd_init= sprintf('dbstop if error ; VP_CODE= ''%s''; TODAY_DIR= ''%s''; set_general_port_fields(''localhost''); general_port_fields.feedback_receiver = ''tobi_c'';  global acquire_func; acquire_func=@acquire_sigserv;', VP_CODE, TODAY_DIR);
if isempty(varargin)
  cmd_bbci= sprintf('storeData(%i, ''localhost'', TODAY_DIR, ''%s'')', fs, final_fname);
else
  OPT= cellfun(@toString, varargin, 'UniformOutput',false);
  cmd_bbci= sprintf('storeData(%i, ''localhost'', TODAY_DIR, ''%s'', %s)', fs, final_fname, vec2str(OPT));
end

%start a new matlab with the storeData command!
system(['matlab -nosplash -r "' cmd_init cmd_bbci '; exit &']);

if nargout==0,
  clear out
end
