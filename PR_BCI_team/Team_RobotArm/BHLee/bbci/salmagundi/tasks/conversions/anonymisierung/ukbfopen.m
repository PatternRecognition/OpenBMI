function HDR = ukbfopen(arg1);
% UKBFOPEN opens signal files for reading and writing and returns 
%       the header information. Only the Nicolet type is supported
%
% HDR = sopen(Filename);
% use sread to get the eeg data
%
% HDR contains the Headerinformation and internal data
%

% This code is extracted from sopen.m of BIOSIG-toolbox http://biosig.sf.net/


% setup

if nargin~=1 | ~ischar(arg1),
  error('incorrect use of ukbfopen');
end

HDR.FileName = arg1;
HDR.FILE.stdout = 1;
HDR.FILE.stderr = 2;

CHAN = 0;
MODE = '';
PERMISSION = 'r';


ReRefMx = [];

% test for type of file 

HDR.TYPE = 'unknown';
HDR.FILE.OPEN = 0;
HDR.FILE.FID  = -1;
HDR.ERROR.status  = 0; 
HDR.ERROR.message = ''; 


% open file and get default values
fid = fopen(HDR.FileName,'rb','ieee-le');
if fid < 0,
	HDR.ERROR.status = -1; 
        HDR.ERROR.message = sprintf('Error GETFILETYPE: file %s not found.\n',HDR.FileName);
        return;
end

[pfad,file,FileExt] = fileparts(HDR.FileName);
if ~isempty(pfad),
  HDR.FILE.Path = pfad;
else
  HDR.FILE.Path = pwd;
end;
HDR.FILE.Name = file;
HDR.FILE.Ext  = char(FileExt(2:length(FileExt)));

fseek(fid,0,'eof');
HDR.FILE.size = ftell(fid);
fseek(fid,0,'bof');

[s,c] = fread(fid,256,'uchar');
if (c == 0),
  s = repmat(0,1,256-c);
elseif (c < 256),
  s = [s', repmat(0,1,256-c)];
else
  s = s';
end;

tmp = 256.^[0:3]*reshape(s(1:20),4,5);

pos1_ascii10 = min(find(s==10));
FLAG.FS3 = any(s==10);
if FLAG.FS3, 
  FLAG.FS3=all((s(4:pos1_ascii10)>=32) & (s(4:pos1_ascii10)<128)); 	% FREESURVER TRIANGLE_FILE_MAGIC_NUMBER
end; 
ss = char(s);

fclose(fid);



fn = fullfile(HDR.FILE.Path, [HDR.FILE.Name '.bni']);
if exist(fn, 'file')
  fid = fopen(fn,'r','ieee-le');
  HDR.Header = char(fread(fid,[1,inf],'uchar'));
  fclose(fid);
end;
if exist(HDR.FileName, 'file')
  fid = fopen(HDR.FileName,'r','ieee-le');
  status = fseek(fid,-4,'eof');
  if status,
    fprintf(2,'Error GETFILETYPE: file %s\n',HDR.FileName); 
    return; 
  end
  datalen = fread(fid,1,'uint32');
  status  = fseek(fid,datalen,'bof');
  HDR.Header = char(fread(fid,[1,inf],'uchar'));
  fclose(fid);
end;
pos_rate = strfind(HDR.Header,'Rate =');
pos_nch  = strfind(HDR.Header,'NchanFile =');

if ~isempty(pos_rate) & ~isempty(pos_nch),
  HDR.SampleRate = str2double(HDR.Header(pos_rate + (6:9)));
  HDR.NS = str2double(HDR.Header(pos_nch +(11:14)));
  HDR.SPR = datalen/(2*HDR.NS);
  HDR.AS.endpos = HDR.SPR;
  HDR.GDFTYP = 3; % int16;
  HDR.HeadLen = 0; 
  HDR.TYPE = 'Nicolet';  
end;



if HDR.ERROR.status, 
  fprintf(HDR.FILE.stderr,'%s\n',HDR.ERROR.message);
  return;
end;

%% Initialization
if ~isfield(HDR,'NS');
        HDR.NS = NaN; 
end;
if ~isfield(HDR,'SampleRate');
        HDR.SampleRate = NaN; 
end;

if ~isfield(HDR,'PhysDim');
        HDR.PhysDim = ''; 
end;
if ~isfield(HDR,'T0');
        HDR.T0 = repmat(nan,1,6);
end;
if ~isfield(HDR,'Filter');
        HDR.Filter.LowPass  = NaN; 
        HDR.Filter.HighPass = NaN; 
end;
if ~isfield(HDR,'FLAG');
        HDR.FLAG = [];
end;
if ~isfield(HDR.FLAG,'FILT')
        HDR.FLAG.FILT = 0; 	% FLAG if any filter is applied; 
end;
if ~isfield(HDR.FLAG,'TRIGGERED')
        HDR.FLAG.TRIGGERED = 0; % the data is untriggered by default
end;
if ~isfield(HDR.FLAG,'UCAL')
        HDR.FLAG.UCAL = ~isempty(strfind(MODE,'UCAL'));   % FLAG for UN-CALIBRATING
end;
if ~isfield(HDR.FLAG,'OVERFLOWDETECTION')
        HDR.FLAG.OVERFLOWDETECTION = isempty(strfind(upper(MODE),'OVERFLOWDETECTION:OFF'));
end; 
if ~isfield(HDR,'EVENT');
        HDR.EVENT.TYP = []; 
        HDR.EVENT.POS = []; 
end;


HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
if HDR.FILE.FID<0,
  return;
end

HDR.FILE.POS  = 0;
HDR.FILE.OPEN = 1; 
HDR.AS.endpos = HDR.SPR;
HDR.AS.bpb = 2*HDR.NS;
HDR.GDFTYP = 'int16';
HDR.HeadLen = 0; 
        
     


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	General Postprecessing for all formats of Header information 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set FLAGS 
HDR.FLAG.UCAL = ~isempty(strfind(MODE,'UCAL'));   % FLAG for UN-CALIBRATING
HDR.FLAG.OVERFLOWDETECTION = isempty(strfind(upper(MODE),'OVERFLOWDETECTION:OFF'));
%if ~isempty(strfind(upper(MODE),'OVERFLOWDETECTION:ON')) & ~isfield(HDR,'THRESHOLD'),
if HDR.FLAG.OVERFLOWDETECTION & ~isfield(HDR,'THRESHOLD'),
%        fprintf(HDR.FILE.stderr,'Warning SOPEN: OVERFLOWDETECTION not supported because of missing THRESHOLD.\n');
end;

% identify type of signal
if HDR.NS>0,
        if ~isfield(HDR,'Label')	
                HDR.Label = [repmat('#',HDR.NS,1),int2str([1:HDR.NS]')];
        elseif isempty(HDR.Label)	
                HDR.Label = [repmat('#',HDR.NS,1),int2str([1:HDR.NS]')];
        else
                HDR.Label = strvcat(HDR.Label);
        end;
        HDR.CHANTYP = repmat(' ',1,HDR.NS);
        tmp = HDR.NS-size(HDR.Label,1);
        %HDR.Label = [HDR.Label(1:HDR.NS,:);repmat(' ',max(0,tmp),size(HDR.Label,2))];
        tmp = reshape(lower([[HDR.Label(1:min(HDR.NS,size(HDR.Label,1)),:);repmat(' ',max(0,tmp),size(HDR.Label,2))],repmat(' ',HDR.NS,1)])',1,HDR.NS*(size(HDR.Label,2)+1));
        HDR.CHANTYP(ceil([strfind(tmp,'eeg'),strfind(tmp,'meg')]/(size(HDR.Label,2)+1))) = 'E'; 
        HDR.CHANTYP(ceil([strfind(tmp,'emg')]/(size(HDR.Label,2)+1))) = 'M'; 
        HDR.CHANTYP(ceil([strfind(tmp,'eog')]/(size(HDR.Label,2)+1))) = 'O'; 
        HDR.CHANTYP(ceil([strfind(tmp,'ecg'),strfind(tmp,'ekg')]/(size(HDR.Label,2)+1))) = 'C'; 
        HDR.CHANTYP(ceil([strfind(tmp,'air'),strfind(tmp,'resp')]/(size(HDR.Label,2)+1))) = 'R'; 
        HDR.CHANTYP(ceil([strfind(tmp,'trig')]/(size(HDR.Label,2)+1))) = 'T'; 
end;

% add trigger information for triggered data
if HDR.FLAG.TRIGGERED & isempty(HDR.EVENT.POS)
	HDR.EVENT.POS = [0:HDR.NRec-1]'*HDR.SPR+1;
	HDR.EVENT.TYP = repmat(hex2dec('0300'),HDR.NRec,1);
	HDR.EVENT.CHN = repmat(0,HDR.NRec,1);
	HDR.EVENT.DUR = repmat(0,HDR.NRec,1);
end;

% apply channel selections to EVENT table
if any(CHAN) & ~isempty(HDR.EVENT.POS) & isfield(HDR.EVENT,'CHN'),	% only if channels are selected. 
	sel = (HDR.EVENT.CHN(:)==0);	% memory allocation, select all general events
	for k = find(~sel'),		% select channel specific elements
		sel(k) = any(HDR.EVENT.CHN(k)==CHAN);
	end;
	HDR.EVENT.POS = HDR.EVENT.POS(sel);
	HDR.EVENT.TYP = HDR.EVENT.TYP(sel);
	HDR.EVENT.DUR = HDR.EVENT.DUR(sel);	% if EVENT.CHN available, also EVENT.DUR is defined. 
	HDR.EVENT.CHN = HDR.EVENT.CHN(sel);
	% assigning new channel number 
	a = zeros(1,HDR.NS);
	for k = 1:length(CHAN),		% select channel specific elements
		a(CHAN(k)) = k;		% assigning to new channel number. 
	end;
	ix = HDR.EVENT.CHN>0;
	HDR.EVENT.CHN(ix) = a(HDR.EVENT.CHN(ix));	% assigning new channel number
end;	

% complete event information - needed by SVIEWER
if ~isfield(HDR.EVENT,'CHN') & ~isfield(HDR.EVENT,'DUR'),  
	HDR.EVENT.CHN = zeros(size(HDR.EVENT.POS)); 
	HDR.EVENT.DUR = zeros(size(HDR.EVENT.POS)); 

	% convert EVENT.Version 1 to 3, currently used by GDF and alpha
	flag_remove = zeros(size(HDR.EVENT.TYP));        
	types  = unique(HDR.EVENT.TYP);
	for k1 = find(bitand(types(:)',hex2dec('8000')));
	        TYP0 = bitand(types(k1),hex2dec('7fff'));
	        TYP1 = types(k1);
	        ix0 = (HDR.EVENT.TYP==TYP0);
	        ix1 = (HDR.EVENT.TYP==TYP1);
	        if sum(ix0)==sum(ix1), 
	                HDR.EVENT.DUR(ix0) = HDR.EVENT.POS(ix1) - HDR.EVENT.POS(ix0);
	                flag_remove = flag_remove | (HDR.EVENT.TYP==TYP1);
	        else 
	                fprintf(2,'Warning SOPEN: number of event onset (TYP=%s) and event offset (TYP=%s) differ\n',dec2hex(TYP0),dec2hex(TYP1));
	        end;
	end
	if any(HDR.EVENT.DUR<0)
	        fprintf(2,'Warning SOPEN: EVENT ONSET later than EVENT OFFSET\n',dec2hex(TYP0),dec2hex(TYP1));
	        HDR.EVENT.DUR(:) = 0
	end;
	HDR.EVENT.TYP = HDR.EVENT.TYP(~flag_remove);
	HDR.EVENT.POS = HDR.EVENT.POS(~flag_remove);
	HDR.EVENT.CHN = HDR.EVENT.CHN(~flag_remove);
	HDR.EVENT.DUR = HDR.EVENT.DUR(~flag_remove);
end;	

% Calibration matrix
if any(PERMISSION=='r') & ~isnan(HDR.NS);
        if isempty(ReRefMx)     % CHAN==0,
                ReRefMx = eye(max(1,HDR.NS));
        end;
        sz = size(ReRefMx);
        if (HDR.NS > 0) & (sz(1) > HDR.NS),
                fprintf(HDR.FILE.stderr,'ERROR SOPEN: to many channels (%i) required, only %i channels available.\n',size(ReRefMx,1),HDR.NS);
                HDR = sclose(HDR);
                return;
        end;
        if isfield(HDR,'Calib')
          HDR.Calib = HDR.Calib*sparse([ReRefMx; zeros(HDR.NS-sz(1),sz(2))]);
          
          HDR.InChanSelect = find(any(HDR.Calib(2:HDR.NS+1,:),2));
          HDR.Calib = sparse(HDR.Calib([1;1+HDR.InChanSelect(:)],:));
          if strcmp(HDR.TYPE,'native')
            HDR.data = HDR.data(:,HDR.InChanSelect);
          end;
        else 
          id = strfind(HDR.Header,'UvPerBit');
          id = id(1)+length('UvPerBit = ');
          id2 = find(HDR.Header(id:end)==10); id2 = id+id2(1)-1;
          scale = str2num(HDR.Header(id:id2-1));

          HDR.Calib= sparse([ones(1,HDR.NS)*0; scale*eye(HDR.NS)]);
          HDR.InChanSelect = 1:HDR.NS;
        end
end;
