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
% $Id: bbci_bet_prepare.m,v 1.12 2008/03/04 14:22:35 neuro_cvs Exp $

global EEG_RAW_DIR general_port_fields

% Required fields:
if any(~ismember({'train_file', 'setup', 'feedback'},fieldnames(bbci))),
  error('bbci must contain fields train_file, setup, and feedback');
end

if ~isfield(bbci, 'player') | isempty(bbci.player),
  bbci.player = 1;
end

file = [REMOTE_RAW_DIR '\' bbci.train_file];
load(file)
bbci.data = uni.dat;
bbci.sf = uni.sf;
%bbci.mrk = uni.mrk;

% Cnt = 
%      clab: {1x43 cell}
%        fs: 100
%     title: 'VPeaa_10_01_13/imag_fbarrow_pcovmeanVPeaa'
%      file: 'C:\Tanis\data\EEG\bbciRaw/VPeaa_10_01_13/imag_fbarrow_pcovmeanVPeaa'
%         x: [155564x43 double]
%         T: 155564

mnt = make_mnt_session2;
Cnt.clab = mnt.clab;
Cnt.fs = bbci.sf;
Cnt.title = bbci.train_file;
Cnt.file = file;
Cnt.x = double(bbci.data);
Cnt.T = length(bbci.data);

mrk_orig.pos = uni.mrk(:,1)';
mrk_orig.desc = uni.mrk(:,2)';
mrk_orig.type = double(uni.mrk(:,2)');
mrk_orig.fs = bbci.sf;

bbci= set_defaults(bbci, 'host', general_port_fields(1).bvmachine, ...
                        'filt', [], ...
                        'clab', '*', ...
                        'impedance_threshold', 50, ...
                        'fs', bbci.sf);

% LOAD THE DATA
if isempty(bbci.train_file)
  %Online Case.. It calls acquire_bv to retrieve the data info
  state = acquire_nirs(bbci.host);
  acquire_nirs('close');
  Cnt = struct('fs',bbci.fs,...
               'x',zeros(0,length(state.clab)));
  Cnt.clab = state.clab;
  da = clock;
  filebase = sprintf('player%d_%d_%d_%d',bbci.player,da(1:3));
  mrk = struct('fs',bbci.fs,...
               'className',{{}},...
               'y',[]);
else
   %Offline version. Train_file is given, so it reads it from HD
%   hdr= eegfile_readBVheader(bbci.train_file); % <----
%   if ~isempty(bbci.impedance_threshold) & isfield(hdr, 'impedances'),
%     ihighimp= find(max(hdr.impedances,[],1)>bbci.impedance_threshold);
%     bbci.clab_highimp = hdr.clab(ihighimp);
%     if ~isempty(ihighimp),
%       fprintf('Channels discarded due to high impedances: %s.\n', vec2str(bbci.clab_highimp));
%     end
%   else
%       ihighimp=[];
%     bbci.clab_highimp= {};
%   end
%   if iscell(bbci.clab) && isequal(bbci.clab{1}, 'not'),
%     clab_load= cat(2, bbci.clab, bbci.clab_highimp);
%   else
%     idx= chanind(hdr, bbci.clab);
%     clab_load= hdr.clab(setdiff(idx,ihighimp));
%   end
% [Cnt, mrk_orig]=eegfile_readBV(bbci.train_file,'fs',bbci.fs,'filt',bbci.filt,'clab',clab_load);
  
  % Cnt contains the info an data form the file: data, name, chanlabels, etc
  % preprocess marker structure (e.g. add markers) if necessary                                
  if isfield(bbci, 'func_mrk')
    if isfield(bbci, 'func_mrk_opts')
      mrk = eval([bbci.func_mrk, '(mrk_orig, bbci.func_mrk_opts)']);
    else
      mrk = eval([bbci.func_mrk, '(mrk_orig)']);
    end
  else

      
    inx1=find(mrk_orig.desc == 1);
    inx2=find(mrk_orig.desc == 2);
    inx=sort([inx1 inx2]');
    mrk.pos=double(mrk_orig.pos(inx));
    mrk.toe=double(mrk_orig.type(inx));
    mrk.className={'left','right'};
    mrk.y(1,mrk.toe==1)=1;
    mrk.y(2,mrk.toe==2)=1;
    mrk.fs=bbci.fs;
    %mrk= mrk_defineClasses(mrk_orig, bbci.classDef);
    mrk_orig.desc = num2cell(mrk_orig.desc);
    mrk_orig.type = num2cell(mrk_orig.type);
    
    
  end
  
  if iscell(bbci.train_file),
    filebase= bbci.train_file{1};
  else
    filebase= bbci.train_file;
  end
  filebase(find(ismember(filebase,'*')))= [];
end
mnt= getElectrodePositions(Cnt.clab);

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
if isfield(mrk, 'className'),
  nclassesrange= [2,length(mrk.className)];
else
  nclassesrange= [2 2];
end
bbci = set_defaults(bbci,'log',1,...
                         'save_name', filebase, ...
                         'maxBufLength',100,...
                         'minDataLength', 40, ...
                         'mrkQueueLength',100,...
                         'maxDataLength', 40000, ...
                         'ringBufferSize', 30000, ...
                         'withclassification', 1, ...
                         'withgraphics', 1, ...
                         'nclassesrange', nclassesrange);

if (isunix & bbci.save_name(1)~='/') | (~isunix & bbci.save_name(2)~=':')
  global ISCBF
  if ~isempty(ISCBF) & ISCBF,
    bbci.save_name = [EEG_RAW_DIR bbci.save_name];
  else
   c_arr = strfind(bbci.save_name,'_');
   c=c_arr(1);
   d = find(ismember('/\', bbci.save_name));
   if ~isempty(d)
    d=d(1);
   else
     % save_name only has underscores.
     d=c_arr(4);
   end
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
       if ~isempty(TMP_DIR)
        bbci.save_name = [TMP_DIR tr];
       else
         warning('TMP_DIR is empty. The classifier will be saved in the current directory.')
       end
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

if exist('opt','var') & ~isempty(opt),
  bbci.calibration_setup.opt= opt;
  clear opt;
end
if exist('stim','var') & ~isempty(stim),
  bbci.calibration_setup.stim= stim;
end

eval(['bbci_setup_' bbci.setup]);
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
  if isfield(mrk, 'className'),
    % Default value for classes: use the minimal allowed number of classes
    bbci.classes = mrk.className(1:bbci.nclassesrange(1));
%  else
%    bbci.classe= 'undefined';
  end
end
bbci_memo.data_reloaded= 1;

% cosmetic issue
if isfield(Cnt,'title'),
  tit= Cnt.title;
  ib= find(tit=='/');
  if length(ib)>1,      %% path, retain only subdir
    tit= tit(ib(end-1)+1:end);
  end
  iu= min(find(tit=='_'));
  sbj= tit(1:iu-1);
  if bbci.player==2,
    sbj= [sbj '(2)'];
  end
  Cnt.short_title= sbj;
else
  Cnt.title = ['player ' num2str(bbci.player)];
  Cnt.short_title = Cnt.title;
end
