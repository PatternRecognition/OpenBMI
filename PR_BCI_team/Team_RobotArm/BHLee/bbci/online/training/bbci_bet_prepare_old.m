%BBCI_BET_PREPARE LOADS THE EEG DATA AND DEFINE SOME IMPORTANT
%DEFAULTS
%
% INPUT FOR THE SCRIPT:
%   Variable 'bbci', struct with fields
%    train_file    a list of train_files
%    classDef      a cell array which defines the classes 
%                  {Tok1,Tok2,...; Name1, Name2,...}
%    setup         name of a setup (e.g. csp, selfpaced,...)
%                  (see directory setups)
%    player        the player number. Optional, default: 1
%    feedback      the name of a feedback
%
% OUTPUT OF THE SCRIPT
%    Cnt           the loaded data
%    mrk           the loaded marker
%    mnt           a suitable montage
%    bbci          copy of input bbci, with additional fields required
%                  for subsequent routines
%
% Guido Dornhege, 07/12/2004
% TODO: extended documentation by Schwaighase
% $Id: bbci_bet_prepare_old.m,v 1.1 2007/08/09 22:19:57 neuro_cvs Exp $
global EEG_RAW_DIR general_port_fields

% Required fields:
if any(~ismember({'train_file', 'setup', 'feedback'},fieldnames(bbci))),
  error('bbci must contain fields train_file, setup, and feedback');
end

if ~isfield(bbci, 'player') | isempty(bbci.player),
  bbci.player = 1;
end

bbci = set_defaults(bbci,'host',general_port_fields(1).bvmachine,...
                         'fs',100);

% LOAD THE DATA
if isempty(bbci.train_file)
  state = acquire_bv(bbci.fs,bbci.host);
  acquire_bv('close');
  Cnt = struct('fs',bbci.fs,...
	       'x',zeros(0,length(state.clab)));
  Cnt.clab = state.clab;
  da = clock;
  filebase = sprintf('player%d_%d_%d_%d',bbci.player,da(1:3));
  mrk = struct('fs',bbci.fs,...
	       'className',{{}},...
	       'y',[]);
else
  if ~iscell(bbci.train_file)
    Cnt= eegfile_loadBV(bbci.train_file, 'fs',bbci.fs);
    mrk= readMarkerTable(bbci.train_file,bbci.fs);
    %  mrk.toe= [1:size(mrk.y,1)]*mrk.y;
    mrk= makeClassMarkers(mrk, bbci.classDef);
    filebase = bbci.train_file;
  else
    Cnt= eegfile_loadBV(bbci.train_file{1}, 'fs',bbci.fs);
    mrk= readMarkerTable(bbci.train_file{1},bbci.fs);
    mrk= makeClassMarkers(mrk, bbci.classDef);
    %  mrk.toe= [1:size(mrk.y,1)]*mrk.y;
    filebase = bbci.train_file{1};
    for i = 2:length(bbci.train_file)
      cnt = eegfile_loadBV(bbci.train_file{i}, 'fs',bbci.fs);
      Mrk = readMarkerTable(bbci.train_file{i},bbci.fs);
      Mrk = makeClassMarkers(Mrk,bbci.classDef);
      %    Mrk.toe= [1:size(Mrk.y,1)]*Mrk.y;
      [Cnt,mrk] = proc_appendCnt(Cnt,cnt,mrk,Mrk);
      clear cnt Mrk;
    end
  end
end
mnt= setElectrodeMontage(Cnt.clab);

if bbci.player==2 & isempty(chanind(Cnt.clab, 'x*'))
  bbci.player = 1;
  warning('no channels for player 2 detected, switch to player 1');
end

if bbci.player==1
  Cnt = proc_selectChannels(Cnt,'not','x*');
elseif bbci.player ==2
  Cnt = proc_selectChannels(Cnt,'x*');
  for i = 1:length(Cnt.clab)
    Cnt.clab{i} = Cnt.clab{i}(2:end);
  end
else
  error('bbci.player must be 1 or 2');
end


% Various options for bbci_bet_apply, 
% see bbci_bet_apply for documentation
bbci = set_defaults(bbci,'log',1,...
                         'save_name', filebase, ...
                         'maxBufLength',100,...
                         'minDataLength', 40, ...
                         'mrkQueueLength',100,...
                         'maxDataLength', 40000, ...
                         'ringBufferSize', 30000, ...
                         'withclassification', 1, ...
                         'withgraphics', 1, ...
                         'nclassesrange', [2,length(mrk.className)]);

if (isunix & bbci.save_name(1)~='/') | (~isunix & bbci.save_name(2)~=':')
  global ISCBF
  if ~isempty(ISCBF) & ISCBF,
    bbci.save_name = [EEG_RAW_DIR bbci.save_name];
  else
   c = strfind(bbci.save_name,'_');
   c=c(1);
   d = find(ismember('/\', bbci.save_name));
   d=d(1);
   str = bbci.save_name(c+1:d-1);
   str(find(str=='_'))= '/';
   if strcmp(str,datestr(date,25)),
     bbci.save_name = [EEG_RAW_DIR bbci.save_name];
   else
     tr = bbci.save_name;
     d = strfind(bbci.save_name,'/');
     tr(d) = '_';
     if isunix
       bbci.save_name = ['/tmp/' tr];
     else
       global TMP_DIR
       bbci.save_name = [TMP_DIR tr];
     end
   end
  end
end

if isfield(bbci,'logfilebase') 
  if (isunix & bbci.logfilebase(1)~='/') | (~isunix & bbci.logfilebase(2)~=':')
    c = strfind(bbci.logfilebase,'_');
    c=c(1);
    d = strfind(bbci.logfilebase,'/');
    d=d(1);
    str = bbci.logfilebase(c+1:d-1);
    str(find(str=='_'))= '/';
    if strcmp(str,datestr(date,25));
      bbci.logfilebase = [EEG_RAW_DIR bbci.logfilebase];
    else
      tr = bbci.logfilebase;
      d = strfind(bbci.logfilebase,'/');
      tr(d) = '_';
      if isunix
        bbci.logfilebase = ['/tmp/' tr];
      else
        bbci.logfilebase = ['c:\eeg_temp\' tr];
      end
    end
  end
end


eval(['setup_' bbci.setup]);
if isfield(bbci,'setup_opts')
  bbci.setup_opts = append_one_struct(bbci.setup_opts,opt);
else
  bbci.setup_opts = opt;
end

% Let's see whether the setup has defined 'nclassesrange'. If yes, take
% it from there, otherwise use default
if isfield(opt, 'nclassesrange'),
  bbci.nclassesrange = opt.nclassesrange;
end


if ~isfield(bbci, 'classes'),
  % Default value for classes: use the minimal allowed number of classes
  bbci.classes = mrk.className(1:bbci.nclassesrange(1));
end


%% cosmetic issue
if isfield(Cnt,'title')
  is= min(find(Cnt.title=='_'));
  sbj= Cnt.title(1:is-1);
  if bbci.player==2,
    sbj= [sbj '(2)'];
  end
  Cnt.short_title= sbj;
else
  Cnt.title = ['player ' num2str(bbci.player)];
  Cnt.short_title = Cnt.title;
end