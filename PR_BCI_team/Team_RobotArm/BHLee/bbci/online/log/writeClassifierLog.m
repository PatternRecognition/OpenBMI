function numb = writeClassifierLog(modus,varargin);
%WRITECLASSIFERLOG logs the output of bbci_bet_apply online
%
% usage:
%  ('init')    numb = writeClassifierLog('init',opt,logdata);
%  ('marker')  writeClassifierLog('marker',ts,toe,desc);
%  ('change')  num = writeClassifierLog('change',ts,logchange);
%  ('cls')     writeClassifierLog('cls',ts,out);
%  ('udp')     writeClassifierLog('udp',ts,udp);
%  ('message') writeClassifierLog('message',ts,str);
%  ('exit')    writeClassifierLog('exit',ts);
%  ('adapt')   writeClassifierLog('adapt',ts,C);
%
% input:
%  opt       a struct with fields:
%            .log             should be done logging?
%            .logfilebase     a basefilename (full path or 
%                             relatively to EEG_RAW_DIR).
%            .szenario        name of the szenario
%      The log_file is saved to basefilename/year_month_day/szenario_number
%      where number is a continuously running number
%  logdata    a struct of important variables to log (e.g. .cont_proc = cont_proc, .opt = opt, .cls = cls,...)
%  logchange  a cell array of the form {varname1,val1,varname2,val2,...}
%  ts         the timestamp
%  toe        the marker token
%  out        the classifier output (cell array)
%  udp        the udp message to sent (numeric array)
%  str        a string to log
%
% output:
%  numb       the continuously running number 
%
% description:
% init: Initialize logfile with name logfile and continous running number. Furthermore variables are saved in a matlab file with continuous running nummber.
% marker: Log got markers
% change: Log changed variables in an additional mat file (new running number)
% cls: Log Classifier Outputs
% udp: Log udp informations
% mess: Log some messages as string
%
% see bbci_bet_apply and adminMarker
%
% Guido Dornhege, 03/12/2005
% TODO: extended documentation by Schwaighase. 
% $Id: writeClassifierLog.m,v 1.7 2008/03/04 14:25:53 neuro_cvs Exp $

persistent fid log_numb logfile

switch modus
 case 'init'
  %INIT
  opt = varargin{1};
  logdata = varargin{2};
  logdata = set_defaults(logdata,'opt',opt);
  opt = set_defaults(opt,'log',1,...
                         'logfilebase','', ...
                         'feedback', '1d');
  if opt.log
    global TODAY_DIR
    if isempty(opt.logfilebase),
      opt.logfilebase= TODAY_DIR;
    else
      [dmy, logfilebase]= fileparts(opt.logfilebase);
      opt.logfilebase= [TODAY_DIR logfilebase];
    end
    
    da =  datevec(now);
    logfile = sprintf('%s/%s_%04d_%02d_%02d_%02d_%02d_%02d_%03d',opt.logfilebase,opt.feedback,da(1:5),floor(da(6)),round(1000*(da(6)-floor(da(6)))));
    
    d = dir([opt.logfilebase '/*.mat']);
    numb_list = [];
    w = warning;
    warning off
    for id = 1:length(d);
      S=load([opt.logfilebase '/' d(id).name],'log_numb');
      if isfield(S,'log_numb')
        numb_list = [numb_list,S.log_numb];
      end
    end
    warning(w);
    
    numb_list = unique(numb_list);
    log_numb = setdiff(1:length(numb_list)+1,numb_list);
    log_numb = min(log_numb);    
    
    fid = fopen([logfile '.log'],'w');
    if fid==-1,
      global LOG_DIR
      warning('could not open %s for writing -> using folder %s', logfile, LOG_DIR);
      [dmy, fn]= fileparts(logfile);
      logfile= [LOG_DIR fn];
      fid= fopen([logfile '.log'],'w');
    end
    fprintf(fid,'Writing logfile starts at %04d_%02d_%02d, %02d:%02d:%02.3f\n',da(:));
    fprintf(fid,'Used feedback: %s\n',opt.feedback);
    
    fprintf(fid,'Continuous file number: %d\n',log_numb);
    
    fprintf(fid,'The variables\n');
    fi = fieldnames(logdata);
    fprintf(fid,' %s',fi{:});
    fprintf(fid,'\n');
    numb = log_numb;
% $$$     while exist(sprintf('%s_%05d_%05d.mat',logfile,log_numb,numb))
% $$$       numb = numb+1;
% $$$     end    

    fprintf(fid,'are saved to %s.mat\n\n',logfile);
    
    for i = 1:length(fi);
      eval(sprintf('%s = logdata.%s;',fi{i},fi{i}));
    end

    save(sprintf('%s.mat',logfile),fi{:},'log_numb');   
 
  else
    fid = 1;
    numb = 0;
  end
 
 case 'marker'
  %MARKER
    ts = varargin{1};
    toe = varargin{2};
    desc = varargin{3};
    if isnumeric(toe)
        fprintf(fid,'Got a marker at timestamp %d: %d, %s\n',ts,toe,desc);
    else
        fprintf(fid,'Got a marker at timestamp %d: %s, %s\n',ts,toe,desc);
    end
  
 case 'change'
  %CHANGE
  ts = varargin{1};
  logchange = varargin{2};
  da = datevec(now);
    fprintf(fid,'\nParameter changes at timestamp %d (%04d_%02d_%02d, %02d:%02d:%02.6f)\n',ts,da(:));

    fprintf(fid,'The following code was evaluated:  \n');
    fprintf(fid,'%s',logchange);
    fprintf(fid,'\n\n');
% $$$     numb = 1;
% $$$     while exist(sprintf('%s_%05d_%05d.mat',logfile,log_numb,numb))
% $$$       numb = numb+1;
% $$$     end    
% $$$     fprintf(fid,'%s_%05d_%05d.mat for details\n\n',logfile,log_numb,numb);
% $$$     
% $$$     save(sprintf('%s_%05d_%05d.mat',logfile,log_numb,numb),logchange);
% $$$     
% $$$     numb = [log_numb,numb];
    numb = log_numb;
 case 'adapt'
  %ADAPT
  ts = varargin{1};
  cls = varargin{2};
    da = datevec(now);
    fprintf(fid,'\nClassifier adapted at timestamp %d (%04d_%02d_%02d, %02d:%02d:%02.6f)\n',ts,da(:));
    %write_struct(fid,cls(1),'cls(1)');  
    fprintf(fid,'cls(1)=%s',toString(cls(1)));
    fprintf(fid,'\n\n');
    % 
 case 'cls'
  %CLASSIFIER_OUTPUTS
  if ~isempty(fid) & fid~=1
    ts = varargin{1};
    out = varargin{2};
    fprintf(fid,'Classifier output calculated at timestamp %d: {',ts);
    for i = 1:length(out)
      if isempty(out{i}),
        fprintf(fid,'NaN');
      elseif length(out{i})==1
        fprintf(fid,'%f',out{i});
      else
        fprintf(fid,'[%f',out{i}(1));
        fprintf(fid,',%f',out{i}(2:end));
        fprintf(fid,']');
      end
      if i<length(out)
        fprintf(fid,',');
      end
    end
    fprintf(fid,'}\n');
  end
  
 case 'udp'
  %UDP COMMUNICATION
  if ~isempty(fid) & fid~=1
    ts = varargin{1};
    udp = varargin{2};
    fprintf(fid,'Send to udp at timestamp %d: ',ts);
    fprintf(fid,'[%f',udp(1));
    if length(udp)>1
        fprintf(fid,',%f',udp(2:end));
    end
    fprintf(fid,']\n');
  end
  
 case 'message'
  %MESSAGES
    ts = varargin{1};
    str = varargin{2};
    fprintf(fid,'Message at timestamp %d: %s\n',ts,str);
  
 case 'exit'
  %EXIT
  ts = varargin{1};
  da = datevec(now);
  if ~isempty(fid) & fid~=1
	  fprintf(fid,'The log-file is finished at timestamp %d (%04d_%02d_%02d, %02d:%02d:%02.6f)\n',ts,da(:));
    fclose(fid);
  end
    
 otherwise
  error('Unknown case');
end

return;

function write_struct(fid,str,prefix)
if isstruct(str)
	% recurse deeper
	f = fieldnames(str);
	for ii=1:length(f)
		write_struct(fid,getfield(str,f{ii}),[prefix '.' f{ii}]);
	end
elseif isnumeric(str)	
	% start printing: str is actually a double
	fprintf(fid,[prefix ': ']);
    fprintf(fid,'%d ', str);
    fprintf(fid,' ;\n ');
end
