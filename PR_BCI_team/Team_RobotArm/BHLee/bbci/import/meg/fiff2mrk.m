function [mrk, triggerChans, miscChans] = fiff2mrk(file, varargin)
%  [mrk] = fiff2mrk(fifname,varargin)
%
%  reads data from .fif file into "cnt"-struct 
%  
%  argin: 
%        file      - string with the name of the .fif including or
%                    without the extension, if starting with '/' a
%                    complete path is assumed to be given, else the
%                    filename is given relative to the MEG_RAW_DIR
%        'fs'      - sampling rate for downsampling the original data          
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
  OUTFILE = file ;  
  
  for i = 1:2:length(varargin) ,
    Param = varargin{i} ;
    Value = varargin{i+1} ;
    if ~isstr(Param), error('Optional parameters must be a string followed by the value');  end ;
    if isstr(Value), Value = lower(Value) ; end ;
    
    switch lower(Param)
     case 'fs'
      FS = Value ;
     case 'outfile'
      OUTFILE = Value ;
    end ;
  end ;
  
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

  rowTRIG = find(ki==3);    nTRIGChans = length(rowTRIG) ;
  rowMISC = find(ki==502);  nMISCChans = length(rowMISC) ;

  
  if nTRIGChans == 1,fprintf('Trigger=%d. ',rowTRIG(1)); end
  if nTRIGChans > 1, fprintf('Trigger=%d to %d. ',rowTRIG(1),rowTRIG(end));  end
  fprintf('\n')

  triggerChans = [] ;
  
  if ~isempty(rowTRIG) ,
    mrk.fs  = sf/dsr ;
    triggerChans = B(rowTRIG,:) ;
  end ;
  if ~isempty(rowMISC) ,
    mrk.fs  = sf/dsr ;
    miscChans = B(rowMISC,:) ;
  end ;
  
  %------------------------------------------------------------------------
  % Read chunks of fif file 
  %------------------------------------------------------------------------
  
  fprintf('Reading data blockwise ...\n');
  while (strcmp(status,'ok') | strcmp(status,'skip')) ,
    [B,status]=rawdata('next');
    while strcmp(status,'skip')
      [B,status]=rawdata('next');
    end
    if (strcmp(status,'ok')) ,
      if ~isempty(rowTRIG) ,
	triggerChans = [triggerChans B(rowTRIG,:) ] ; 
      end ;
      if ~isempty(rowMISC) ,
	miscChans = [miscChans B(rowMISC,:) ] ; 
      end ;
    end ;
  end ;

%  [mrk.pos, mrk.toe] = getTrigger(triggerChans) ;

  [mrk.pos, mrk.toe] = getTrigger(miscChans) ;
  rawdata('close') ;

  % remove all triggers with a ISI below 10 ms
%  rmPos = find(diff(mrk.pos)< (10/1000*sf)) ;
%  mrk.pos([rmPos+1]) = [] ;
%  mrk.toe([rmPos+1]) = [] ;

% compensate for a bug in SEPtrigger.exe 
  % where at the end of each block a toe==2 is send
  if ( (mrk.toe(end) == 2) & (sum(mrk.toe==2)==1) ),
    mrk.toe(end) = [] ;
    mrk.pos(end) = [] ;
  end ;

  % set unique trigger channels
  nPos  = length(mrk.pos) ;
  triggers = unique(mrk.toe(mrk.toe~=0)) ;
  nTriggers= length(triggers) ;
  mrk.y = zeros(nTriggers, nPos) ;
  for trigger = 1: nTriggers ,
    mrk.className{trigger} = sprintf('Trigger #%d',triggers(trigger)) ;
    trigIdx = find(mrk.toe == triggers(trigger)) ;
    mrk.y(trigger,trigIdx) = 1 ;
  end ;

  mkr.pos = floor(mrk.pos /dsr +0.5) ;
  
  fullName = [MEG_MAT_DIR OUTFILE  '_MRK']
  [filepath, filename]= fileparts(fullName);
  if ~exist(filepath, 'dir'),
    [parentdir, newdir]=fileparts(filepath);
    [status,msg]= mkdir(parentdir, newdir);
    if status~=1,
      error(msg);
    end
  end
    
  % save the marker-file and the montage
  save(fullName, 'mrk');


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
  
  triggerChans = [zeros(nTrig,1) diff(triggerChans,1,2) ] ;
  
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
  



















