function generate_subject_setup(subject, varargin)
%GENERATE_SUBJECT_SETUP - Generates Subject Setups for BBCI Feedback
%
%This functions generates matlab scripts that can be used as
%subject setup files. Those files are saved in
%  [neuro_cvs]/bci/bbci_bet/subjects/
%and are automatically checked into the CVS. There are also
%folders created for the EEG data. These folders are created in
%EEG_RAW_DIR (global variable) and permission are set to be
%group writable.
%
%Synopsis:
% generate_subject_setup(SUBJECT, <OPT>)
%
%Arguments:
% SUBJECT: This can a string (name resp. code) of the subject, or
%    a cell array of two such strings for the two subject mode. If
%    the latter case the first subject is assumed to be player one.
% OPT: struct or property/value list of optional properties
%  .player - define the player number (1 or 2) for two player mode
%  .root_sbj - name of the subject
%
%The optional properties are mainly of internal use. In the typical
%case they are not needed.
%
%Examples:
% generate_subject_setup('Klaus')
% generate_subject_setup({'VPcx','VPcy'})


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'player', 1, ...
                  'root_sbj', subject);

global BBCI_DIR EEG_RAW_DIR
SBJ_SETUP_DIR= [BBCI_DIR 'subjects/'];

if iscell(subject),
  for cc= 1:length(subject);
    generate_subject_setup(subject{cc}, opt, 'player',cc, ...
                            'root_sbj',subject{1});
  end
  return;
end

today_str= datestr(now, 'yy/mm/dd');
today_str(find(today_str=='/'))= '_';
sub_dir= [subject '_' today_str];
file= [ sub_dir '.m'];
%SBJ_SETUP_DIR
if ~exist(file,'file')
  fid= fopen(file, 'wt');
  if fid==-1,
    error(sprintf('cannot open file <%s> for writing', file));
  end
  
  fprintf(fid, ['bbci.train_file= {''' opt.root_sbj '_' today_str ...
		'/imag_lett' opt.root_sbj '''};\n']);
  fprintf(fid, 'bbci.classDef = {1,2,3;''left'',''right'',''foot''};\n');
  fprintf(fid, 'bbci.player = %d;\n', opt.player);
  fprintf(fid, 'bbci.setup = ''csp'';\n');
  fprintf(fid, ['bbci.save_name = ''' sub_dir '/imag_' subject ''';\n']);
  fprintf(fid, 'bbci.feedback = ''1d'';\n');
  fprintf(fid, 'bbci.classes = {''left'',''right''};\n');
  
  fclose(fid);
  
  %unix(sprintf('cd %s; cvs add %s.m; cvs commit -m "" %s.m', SBJ_SETUP_DIR, ...
	      % sub_dir, sub_dir));
end
% also generate the data folders, if they do not yet exist.
eeg_dir = [EEG_RAW_DIR sub_dir];
if isunix & ~exist(eeg_dir,'dir'),
  unix(sprintf('mkdir %s;chmod g=u %s;',eeg_dir,eeg_dir));
end