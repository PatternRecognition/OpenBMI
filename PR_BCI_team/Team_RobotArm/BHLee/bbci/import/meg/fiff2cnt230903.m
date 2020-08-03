function [cntEEG, cntMEG, mntEEG, mntMEG, mrk] = fiff2cnt230903(file, varargin)
%  [cnt, mrk, mnt] = fiff2cnt(fifname,varargin)
%
%  reads data from .fif file into "cnt"-struct 
%  
%  argin: 
%        file      - string with the name of the .fif including or
%                    without the extension, if starting with '/' a
%                    complete path is assumed to be given, else the
%                    filename is given relative to the MEG_RAW_DIR
%        'fs'      - sampling rate for downsampling the original data          
%        'chans'   - valid values are {MAG,GRAD1,GRAD2,GRAD12,EEG,MEG}
%        'outfile' - name of the matlab file to store the processed files
%
  
  global MEG_RAW_DIR MEG_CFG_DIR MEG_MAT_DIR EEG_MAT_DIR

  % generate fif-filename and test for the existence of the file
  if (file(1) ~= filesep),
    fifname = [MEG_RAW_DIR file] ;
  else 
    fifname = file ;
  end;
  if ~(any(findstr(file,'.')== (length(file)-3)))
    fifname = [fifname '.fif'] ;
  else 
    file((end-3):end) = [] ;
  end ;
  
  if ~exist(fifname,'file')
    fprintf(sprintf('>>>>>>> Error, file not found, please check path : %s\n',fifname)) ;
    return ;
  end ;
  
  if (rem(length(varargin) ,2)),
    error('Number of optional arguments is not valid.\n')
  end ;
  
  % set the defaultvalues
  CHANS   = 'all' ;
  OUTFILE = file ;  
  
  for i = 1:2:length(varargin) ,
    Param = varargin{i} ;
    Value = varargin{i+1} ;
    if ~isstr(Param), error('Optional parameters must be a string followed by the value');  end ;
    if isstr(Value), Value = lower(Value) ; end ;
    
    switch lower(Param)
     case 'chans'
      CHANS = Value ;
     case 'fs'
      FS = Value ;
     case 'outfile'
      OUTFILE = Value ;
    end ;
  end ;
  
  cntEEG = [] ;
  cntMEG = [] ;
  mntEEG = [] ;
  mntMEG = [] ;
  
  rawdata('any',fifname);
  sf = rawdata('sf'); 
  if exist('FS','var'), dsr = round(sf/FS) ; else dsr = 1; end ;
  T  = rawdata('samples') ;
  Tds= 1 + floor((T-1)/dsr) ;
  
  [B,status]=rawdata('next');
  while strcmp(status,'skip') ,
    [B,status]=rawdata('next');
  end ;
  [nChans, TB] = size(B) ;
  
  [na,ki,nu]=channames(fifname);
  na = strrmspace(na) ;

  rowMEG  = [] ; 
  rowEOG  = [] ;
  rowMISC = [] ;
  rowTRIG = [] ;
  SSPtrans =[] ;

  rowMEG = find(ki==1);     nMEGChans  = length(rowMEG) ;
  rowEEG = find(ki==2);     nEEGChans  = length(rowEEG) ;
  rowEOG = find(ki==202);   nEOGChans  = length(rowEOG) ;
  rowEMG = find(ki==302);   nEMGChans  = length(rowEMG) ;
  rowTRIG = find(ki==3);    nTRIGChans = length(rowTRIG) ;
  rowMISC = find(ki==502);  nMISCChans = length(rowMISC) ;

  fprintf('Data matrix, rows: ');
  if length(rowMEG) == 1
    fprintf('MEG=%d. ',rowMEG(1));
  end
  if length(rowMEG) > 1,  fprintf('MEG=%d to %d. ',rowMEG(1),rowMEG(end)); end
  if length(rowEOG) == 1, fprintf('EOG=%d. ',rowEOG(1));  end
  if length(rowEOG) > 1,  fprintf('EOG=%d to %d. ',rowEOG(1),rowEOG(end));  end
  if length(rowTRIG) == 1,fprintf('Trigger=%d. ',rowTRIG(1)); end
  if length(rowTRIG) > 1, fprintf('Trigger=%d to %d. ',rowTRIG(1),rowTRIG(end));  end
  if length(rowMISC) == 1,fprintf('Misc=%d. ',rowMISC(1)); end
  if length(rowMISC) > 1, fprintf('Misc=%d to %d. ',rowMISC(1),rowMISC(end)); end
  if length(rowEEG) == 1, fprintf('EEG=%d. ',rowEEG(1)); end
  if length(rowEEG) > 1,  fprintf('EEG=%d to %d. ',rowEEG(1),rowEEG(end));end
  if length(rowEMG) == 1, fprintf('EMG=%d. ',rowEMG(1));end
  if length(rowEMG) > 1,  fprintf('EMG=%d to %d. ',rowEMG(1),rowEMG(end));end
  fprintf('\n')
  
  badchanlist = []; 
  if length(rowMEG) == 306
    % For Neuromag306 get the SSP transformation
    SSPtrans = projmat([MEG_CFG_DIR 'Emptyroom_avg.fif']) ;
    SSPtrans = SSPtrans *1e12 ;
    megmodel([0 0 0],fifname);
    % megmodel('device',[0 0 0],fifname);
    [badchanlist,badchannames] = badchans;
  elseif length(rowMEG) == 122
    megmodel([0 0 0],fifname);
    % megmodel('device',[0 0 0],fifname);
    [badchanlist,badchannames] = badchans;
  else
   % EEG DATA
  end

%   BADCHtrans = zeros(length(rowMEG));
%   for k=1:length(rowMEG)
%     BADCHtrans(k,k) = 1;
%   end
  BADCHtrans = eye(length(rowMEG)) ;
  badchanlist = find(badchanlist == 1);
  for k=1:length(badchanlist)
    BADCHtrans(badchanlist(k),badchanlist(k)) = 0 ;
  end

  actT = 0 ;
  actTds = 0 ;
  startB = 1 ;

  subsamples   = startB:dsr:TB ;
  TBds         = length(subsamples) ;
  if (~isempty(rowEEG)) & (strcmp(CHANS,'eeg') | (strcmp(CHANS,'all'))) ,
    cntEEG.clab        = na([rowEEG; rowEOG]) ;
    cntEEG.x           = zeros(Tds, nEEGChans+nEOGChans) ;
    cntEEG.x(actTds+ (1:TBds),:) = 1e6*B([rowEEG; rowEOG],subsamples)' ;
    cntEEG.title       = [file] ;
    cntEEG.fs          = sf/dsr ;
  end ;

%  if (~isempty(rowMEG) & (strcmp(CHANS,'meg'))| (strcmp(CHANS,'all')) ),
  if (~isempty(rowMEG) & any(strcmp(CHANS,{'meg','all','grad1','grad2','mag'}))),

    B(rowMEG,:) = SSPtrans *B(rowMEG,:) ;
    switch CHANS,
     case 'grad1'
      cntMEG.clab        = na([rowMEG]) ;
      cntMEG.x           = zeros(Tds, (nMEGChans/3)) ;
      cntMEG.x(actTds+(1:TBds),:) = B(rowMEG(1:3:nMEGChans),subsamples)' ;
     case 'grad2'
      cntMEG.clab        = na([rowMEG]) ;
      cntMEG.x           = zeros(Tds, (nMEGChans/3)) ;
      cntMEG.x(actTds+(1:TBds),:) = B(rowMEG(2:3:nMEGChans),subsamples)' ;
     case 'mag'
      cntMEG.clab        = na([rowMEG; rowEOG]) ;
      cntMEG.x           = zeros(Tds, (nMEGChans/3)+nEOGChans) ;
      cntMEG.x(actTds+(1:TBds),:) = B([rowMEG(3:3:nMEGChans); rowEOG],subsamples)' ;
     otherwise
      cntMEG.clab        = na([rowMEG; rowEOG]) ;
      cntMEG.x           = zeros(Tds, nMEGChans+nEOGChans) ;
      cntMEG.x(actTds+(1:TBds),:) = B([rowMEG; rowEOG],subsamples)' ;
    end ;
    cntMEG.title       = [file] ;
    cntMEG.fs          = sf/dsr ;
  end;
  if ~isempty(rowTRIG) ,
    mrk.fs  = sf/dsr ;
    [mrk.pos, mrk.toe] = getTrigger(B(rowTRIG,:)) ;
  end ;
  
  %------------------------------------------------------------------------
  % Read chunks of fif file 
  %------------------------------------------------------------------------
  
  fprintf('Reading data blockwise ...\n');
  while (strcmp(status,'ok') | strcmp(status,'skip')) ,
    actTds = actTds +TBds ;
    actT = actT +TB ;
    startB = subsamples(end) + dsr - TB ;
    [B,status]=rawdata('next');
    while strcmp(status,'skip')
      [B,status]=rawdata('next');
    end

    if (strcmp(status,'ok')) ,
      [nChans, TB] = size(B) ;
      subsamples   = startB:dsr:TB ;
      TBds         = length(subsamples) ;

      if (~isempty(rowEEG)) & (strcmp(CHANS,'eeg') | (strcmp(CHANS,'all'))) ,
	cntEEG.x(actTds+(1:TBds),:) = 1e6*B([rowEEG; rowEOG],subsamples)' ;
      end ;
      
      if (~isempty(rowMEG) & any(strcmp(CHANS,{'meg','all','grad1','grad2','mag'}))),

	B(rowMEG,:) = SSPtrans *B(rowMEG,:) ;
	switch CHANS,
	 case 'grad1'
	  cntMEG.clab        = na([rowMEG(1:3:nMEGChans)]) ;
	  cntMEG.x(actTds+(1:TBds),:) = B(rowMEG(1:3:nMEGChans),subsamples)' ;
	 case 'grad2'
	  cntMEG.clab        = na([rowMEG(2:3:nMEGChans)]) ;
	  cntMEG.x(actTds+(1:TBds),:) = B(rowMEG(2:3:nMEGChans),subsamples)' ;
	 case 'mag'
	  cntMEG.clab        = na([rowMEG(3:3:nMEGChans); rowEOG]) ;
	  cntMEG.x(actTds+(1:TBds),:) = B([rowMEG(3:3:nMEGChans); rowEOG],subsamples)' ;
	 otherwise
	  cntMEG.clab        = na([rowMEG; rowEOG]) ;
	  cntMEG.x(actTds+(1:TBds),:) = B([rowMEG; rowEOG],subsamples)' ;
	end ;
      end ;
    
      if ~isempty(rowTRIG) ,
	[pos, toe] = getTrigger(B(rowTRIG,:)) ;
	mrk.pos = [mrk.pos (pos+actT)] ;
	mrk.toe = [mrk.toe toe] ;
      end ;
      

      T = T + size(B,2) ;
    
    end ;
  end ;
  
  rawdata('close') ;

  % remove all triggers with a ISI below 10 ms
  rmPos = find(diff(mrk.pos)< (10/1000*sf)) ;
  mrk.pos([rmPos+1]) = [] ;
  mrk.toe([rmPos+1]) = [] ;
  % compensate for a bug in SEPtrigger.exe 
  % where at the end of each block a toe==2 is send
  if ( (mrk.toe(end) == 2) & (sum(mrk.toe==2)==1) ),
    mrk.toe(end) = [] ;
    mrk.pos(end) = [] ;
  end ;
  % remove empty trigger channels
  nPos  = length(mrk.pos) ;
  triggers = unique(mrk.toe(mrk.toe>0)) ;
  nTriggers= length(triggers) ;
  mrk.y = zeros(nTriggers, nPos) ;
  for trigger = 1: nTriggers ,
    mrk.className{trigger} = sprintf('Trigger #%d',triggers(trigger)) ;
    trigIdx = find(mrk.toe == triggers(trigger)) ;
    mrk.y(trigger,trigIdx) = 1 ;
  end ;

  mkr.pos = floor(mrk.pos /dsr +0.5) ;
  
  if ~isempty(cntEEG),
    % remove bad eeg-channels (EEG029, EEG031)
    cntEEG = proc_selectChannels(cntEEG, {'not','EEG029','EEG031'}) ;
    cntEEG.clab = {'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT9','FT7','FC5','FC1','FC2','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP3','CP1','CP2','CP4','TP8','TP10','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','PO4','PO8','O1','Oz','O2','Iz','EOG1','EOG2'};
    mntEEG = eegMontageHUT ;
    saveProcessedEEG([EEG_MAT_DIR OUTFILE], cntEEG, mrk, mntEEG,'EEG') ;
  end ;
  
  if ~isempty(cntMEG),
    clab = na([rowMEG; rowEOG]) ;
%    clab([(end-1):end]) = {'EOG1','EOG2'} ;
    mntMEG = setMEGMontage(clab);

    fullName = [MEG_MAT_DIR OUTFILE  '_MEG']
    [filepath, filename]= fileparts(fullName);
    if ~exist(filepath, 'dir'),
      [parentdir, newdir]=fileparts(filepath);
      [status,msg]= mkdir(parentdir, newdir);
      if status~=1,
	error(msg);
      end
    end
    
    mnt = mntMEG ;
    % save the marker-file and the montage
    
    if ~exist([fullName '.mat'], 'file'),
      save(fullName, 'mrk','mnt');
    end

    % save the data channelwise into a matlab-file, using the channel-Label
    fprintf('saving channel:');
    for channel = 1: length(cntMEG.clab),
      eval([cntMEG.clab{channel} '= cntMEG.x(:,channel);']) ;
      fprintf(sprintf('%s,',cntMEG.clab{channel})) ;
      save(fullName, cntMEG.clab{channel},'-append') ;
      clear(cntMEG.clab{channel}) ;
    end ;
    fprintf('\n');
  end ;
  
  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% SUBFUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[pos, toe] = getTrigger(triggerChans) 

  % set Threshold for trigger amplitude
  TrigThres = 2 ; 
  
  % get the dimensions
  [nTrig, T] = size(triggerChans) ;
  
  % thresholding the trigger channels
  triggerChans = triggerChans > TrigThres ;
  
  % binary coding of the Type Of Event (TOE)
  toe = 2.^[0:(nTrig-1)] *triggerChans ;
  
  % get the index of the non-zero events
  pos = find(toe~=0) ;
  
  % remove all zero events
  toe(toe==0) = [] ;
  
  % remove all triggers with a ISI below 5 samples 
%  rmPos = find(diff(pos)< 5) ;
%  pos([rmPos+1]) = [] ;
%  toe([rmPos+1]) = [] ;
  



















