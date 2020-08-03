function [HDR,H1,h2] = sopen(arg1,PERMISSION,CHAN,MODE,arg5,arg6)
% SOPEN opens signal files for reading and writing and returns 
%       the header information. Many different data formats are supported.
%
% HDR = sopen(Filename, PERMISSION, [, CHAN [, MODE]]);
% [S,HDR] = sread(HDR, NoR, StartPos);
% HDR = sclose(HDR);
%
% PERMISSION is one of the following strings 
%	'r'	read header
%	'w'	write header
%
% CHAN defines a list of selected Channels
%   Alternative CHAN can be also a Re-Referencing Matrix ReRefMx
%       (i.e. a spatial filter). 
%   E.g. the following command returns the difference and 
%       the mean of the first two channels. 
%   HDR = sopen(Filename, 'r', [[1;-1],[.5,5]]);
%   [S,HDR] = sread(HDR, Duration, Start);
%   HDR = sclose(HDR);
%
% MODE  'UCAL'  uncalibrated data
%       'OVERFLOWDETECTION:OFF' turns off automated overflow detection
%       Several options can be concatenated within MODE. 
%
% HDR contains the Headerinformation and internal data
% S 	returns the signal data 
%
% Several files can be loaded at once with SLOAD
%
% see also: SLOAD, SREAD, SSEEK, STELL, SCLOSE, SWRITE, SEOF


% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

%	$Revision: 1.119 $
%	$Id: sopen.m,v 1.119 2005/09/10 20:43:54 schloegl Exp $
%	(C) 1997-2005 by Alois Schloegl <a.schloegl@ieee.org>	
%    	This is part of the BIOSIG-toolbox http://biosig.sf.net/


if isnan(str2double('1, 3'));
        fprintf(2,'Warning BIOSIG: incorrect version of STR2DOUBLE.\n');
        fprintf(2,'- Its recommended to update STR2DOUBLE. Contact Alois!\n');
end;

global FLAG_NUMBER_OF_OPEN_FIF_FILES;

if ischar(arg1),
        HDR.FileName = arg1;
        HDR.FILE.stdout = 1;
        HDR.FILE.stderr = 2;
%elseif length(arg1)~=1,
%	HDR = [];
elseif isfield(arg1,'name')
        HDR.FileName = arg1.name;
	HDR.FILE = arg1; 
        HDR.FILE.stdout = 1;
        HDR.FILE.stderr = 2;
else %if isfield(arg1,'FileName')
        HDR = arg1;
%else
%	HDR = [];
end;

if ~isfield(HDR,'FILE'),
        HDR.FILE.stdout = 1;
        HDR.FILE.stderr = 2;
end;	
if ~isfield(HDR.FILE,'stdout'),
        HDR.FILE.stdout = 1;
end;	
if ~isfield(HDR.FILE,'stderr'),
        HDR.FILE.stderr = 2;
end;

if nargin<3, CHAN = 0; end; 
if nargin<4, MODE = ''; end;
if nargin<2, 
        PERMISSION='r'; 
elseif isempty(PERMISSION),
        PERMISSION='r'; 
elseif isnumeric(PERMISSION),
        fprintf(HDR.FILE.stderr,'Warning SOPEN: second argument should be PERMISSION, assume its the channel selection\n');
        CHAN = PERMISSION; 
        PERMISSION = 'r'; 
elseif ~any(PERMISSION(1)=='RWrw'),
        fprintf(HDR.FILE.stderr,'Warning SOPEN: PERMISSION must be ''r'' or ''w''. Assume PERMISSION is ''r''\n');
        PERMISSION = 'r'; 
end;
if ~any(PERMISSION=='b');
        PERMISSION = [PERMISSION,'b']; % force binary open. Needed for Octave
end;

if all(size(CHAN)>1) | any(floor(CHAN)~=CHAN) | (any(CHAN<0) & (numel(CHAN)>1));
        ReRefMx = CHAN; 
        CHAN = find(any(CHAN,2));
elseif all(CHAN>0),
	if any(diff(CHAN)<=0),
	%	fprintf(HDR.FILE.FID,'Warning SOPEN: CHAN-argument not sorted - header information like Labels might not correspond to data.\n');
	end;	
        ReRefMx = sparse(CHAN,1:length(CHAN),1);
else    
        ReRefMx = [];
end
if isempty(MODE), MODE=' '; end;	% Make sure MODE is not empty -> FINDSTR

% test for type of file 
if any(PERMISSION=='r'),
        HDR = getfiletype(HDR);
	if HDR.ERROR.status, 
		fprintf(HDR.FILE.stderr,'%s\n',HDR.ERROR.message);
		return;
	end;
else
	[pfad,file,FileExt] = fileparts(HDR.FileName);
	HDR.FILE.Name = file;
	HDR.FILE.Path = pfad;
	HDR.FILE.Ext  = FileExt(2:length(FileExt));
	HDR.FILE.OPEN = 0;
        HDR.FILE.FID  = -1;
	HDR.ERROR.status  = 0; 
	HDR.ERROR.message = ''; 
end;

%% Initialization
if ~isfield(HDR,'NS');
        HDR.NS = NaN; 
end;
if ~isfield(HDR,'SampleRate');
        HDR.SampleRate = NaN; 
end;
if 0, ~isfield(HDR,'Label');
        HDR.Label = []; 
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
%%%%% Define Valid Data types %%%%%%
%GDFTYPES=[0 1 2 3 4 5 6 7 16 17 255+(1:64) 511+(1:64)];
GDFTYPES=[0 1 2 3 4 5 6 7 16 17 18 255+[1 12 22 24] 511+[1 12 22 24]];

%%%%% Define Size for each data type %%%%%
GDFTYP_BYTE=zeros(1,512+64);
GDFTYP_BYTE(256+(1:64))=(1:64)/8;
GDFTYP_BYTE(512+(1:64))=(1:64)/8;
GDFTYP_BYTE(1:19)=[1 1 1 2 2 4 4 8 8 4 8 0 0 0 0 0 4 8 16]';

if 0, strcmp(HDR.TYPE,'EDF') | strcmp(HDR.TYPE,'GDF') | strcmp(HDR.TYPE,'BDF'),
        if any(PERMISSION=='w');
                HDR = eegchkhdr(HDR);
        end;
        if nargin<4,
                HDR = sdfopen(HDR,PERMISSION);
        else
                HDR = sdfopen(HDR,PERMISSION,CHAN,MODE);
        end;

	%%% Event file stored in GDF-format
	if ~any([HDR.NS,HDR.NRec,~length(HDR.EVENT.POS)]);
		HDR.TYPE = 'EVENT';
	end;	
        
        
elseif strcmp(HDR.TYPE,'EDF') | strcmp(HDR.TYPE,'GDF') | strcmp(HDR.TYPE,'BDF'),
        H1idx = [8 80 80 8 8 8 44 8 8 4];
        H2idx = [16 80 8 8 8 8 8 80 8 32];
        
        HDR.ErrNo = 0; 
        HANDEDNESS = {'unknown','right','left','equal'}; 
        GENDER  = {'unknown','male','female'};
        SCALE13 = {'unknown','no','yes'};
        SCALE14 = {'unknown','no','yes','corrected'};
                
        if any(PERMISSION=='r');
                [HDR.FILE.FID,MESSAGE]=fopen(HDR.FileName,'r','ieee-le');          
                
                if HDR.FILE.FID<0 
                        HDR.ErrNo = [32,HDR.ErrNo];
                        return;
                end;

                %%% Read Fixed Header %%%
                [H1,count]=fread(HDR.FILE.FID,[1,192],'uchar');     %
                if count<192,
                        HDR.ErrNo = [64,HDR.ErrNo];
                        return;
                end;
                
                HDR.VERSION=char(H1(1:8));                     % 8 Byte  Versionsnummer 
                if ~(strcmp(HDR.VERSION,'0       ') | all(abs(HDR.VERSION)==[255,abs('BIOSEMI')]) | strcmp(HDR.VERSION(1:3),'GDF'))
                        HDR.ErrNo = [1,HDR.ErrNo];
                        if ~strcmp(HDR.VERSION(1:3),'   '); % if not a scoring file, 
                                %	    return; 
                        end;
                end;
                if strcmp(char(H1(1:8)),'0       ') 
                        HDR.VERSION = 0; 
                elseif all(abs(H1(1:8))==[255,abs('BIOSEMI')]), 
                        HDR.VERSION = -1; 
                elseif all(H1(1:3)==abs('GDF'))
                        HDR.VERSION = str2double(char(H1(4:8))); 
                else
                        HDR.ErrNo = [1,HDR.ErrNo];
                        if ~strcmp(HDR.VERSION(1:3),'   '); % if not a scoring file, 
                                %	    return; 
                        end;
                end;
                
		HDR.Patient.Sex = 0;
		HDR.Patient.Handedness = 0;
if 0,
		HDR.Patient.Weight = NaN;
		HDR.Patient.Height = NaN;
		HDR.Patient.Impairment.Visual = NaN;
		HDR.Patient.Smoking = NaN;
		HDR.Patient.AlcoholAbuse = NaN;
		HDR.Patient.DrugAbuse = NaN;
		HDR.Patient.Medication = NaN;
end;
                %if strcmp(HDR.VERSION(1:3),'GDF'),
                if strcmp(HDR.TYPE,'GDF'),
                        if (HDR.VERSION >= 1.90)
                                HDR.PID = deblank(char(H1(9:84)));                  % 80 Byte local patient identification
                                HDR.RID = deblank(char(H1(89:156)));                % 80 Byte local recording identification
                                [HDR.Patient.Id,tmp] = strtok(HDR.PID,' ');
                                HDR.Patient.Name = tmp(2:end); 
                                
                                HDR.Patient.Medication   = SCALE13{bitand(floor(H1(85)/64),3)+1};
                                HDR.Patient.DrugAbuse    = SCALE13{bitand(floor(H1(85)/16),3)+1};
                                HDR.Patient.AlcoholAbuse = SCALE13{bitand(floor(H1(85)/4),3)+1};
                                HDR.Patient.Smoking      = SCALE13{bitand(H1(85),3)+1};
                                tmp = abs(H1(86:87)); tmp(tmp==0) = NaN; tmp(tmp==255) = inf;
				HDR.Patient.Weight = tmp(1);
				HDR.Patient.Height = tmp(2);
				HDR.Patient.Sex = GENDER{bitand(H1(88),3)+1};
				HDR.Patient.Handedness = HANDEDNESS{bitand(floor(H1(88)/4),3)+1};
                                HDR.Patient.Impairment.Visual = SCALE14{bitand(floor(H1(88)/16),3)+1};
                                if H1(156)>0, 
                                        HDR.RID = deblank(char(H1(89:156)));
                                else
                                        HDR.RID = deblank(char(H1(89:152)));
                                        %HDR.REC.LOC.RFC1876  = 256.^[0:3]*reshape(H1(153:168),4,4);
                                        HDR.REC.LOC.Version   = abs(H1(156));
                                        HDR.REC.LOC.Size      = dec2hex(H1(155));
                                        HDR.REC.LOC.HorizPre  = dec2hex(H1(154));
                                        HDR.REC.LOC.VertPre   = dec2hex(H1(153));
                                end;
                                HDR.REC.LOC.Latitude  = H1(157:160)*256.^[0:3]'/3600000;
				HDR.REC.LOC.Longitude = H1(161:164)*256.^[0:3]'/3600000;
				HDR.REC.LOC.Altitude  = H1(165:168)*256.^[0:3]'/100;

                                tmp = H1(168+[1:16]);
				% little endian fixed point number with 32 bits pre and post comma 
				t1 = tmp(1:8 )*256.^[-4:3]';
				HDR.T0 = datevec(t1);
				t2 = tmp(9:16)*256.^[-4:3]';
				HDR.Patient.Birthday = datevec(t2);
				if (t2 > 1) & (t2 < t1),
					HDR.Patient.Age = floor((t1-t2)/365.25);
				end;	
                                HDR.ID.Equipment = fread(HDR.FILE.FID,[1,8],'uchar');   
                                tmp = fread(HDR.FILE.FID,[1,6],'uint8');
                                HDR.REC.IPaddr = tmp(6:-1:1); 
                                tmp = fread(HDR.FILE.FID,[1,3],'uint16'); 
                                tmp(tmp==0)=NaN;
                                HDR.Patient.Headsize = tmp;
                                tmp = fread(HDR.FILE.FID,[3,2],'float32');
                                HDR.ELEC.REF = tmp(:,1)';
                                HDR.ELEC.GND = tmp(:,2)';
                        else
                                HDR.PID = deblank(char(H1(9:88)));                  % 80 Byte local patient identification
                                HDR.RID = deblank(char(H1(89:168)));                % 80 Byte local recording identification
                                [HDR.Patient.Id,tmp] = strtok(HDR.PID,' ');
                                HDR.Patient.Name = tmp(2:end); 
                                
                                tmp = repmat(' ',1,22);
    		                tmp([1:4,6:7,9:10,12:13,15:16,18:21]) = char(H1(168+[1:16]));
            		        HDR.T0(1:6)   = str2double(tmp);
                    		HDR.T0(6)     = HDR.T0(6)/100;
                                HDR.reserved1 = fread(HDR.FILE.FID,[1,8*3+20],'uchar');   % 44 Byte reserved
                                HDR.ID.Equipment  = HDR.reserved1(1:8);
                    		HDR.ID.Lab        = HDR.reserved1(9:16);
                                HDR.ID.Technician = HDR.reserved1(17:24);
                        end;
			
                        %if str2double(HDR.VERSION(4:8))<0.12,
                        if (HDR.VERSION < 0.12),
                                HDR.HeadLen  = str2double(H1(185:192));    % 8 Byte  Length of Header
                        else
                                HDR.HeadLen  = H1(185:188)*256.^[0:3]';    % 8 Byte  Length of Header
                                HDR.reserved = H1(189:192);
                        end;

                        %HDR.NRec = fread(HDR.FILE.FID,1,'int64');     % 8 Byte # of data records
                        HDR.NRec = fread(HDR.FILE.FID,1,'int32');      % 8 Byte # of data records
                        fread(HDR.FILE.FID,1,'int32');      % 8 Byte # of data records
                        %if strcmp(HDR.VERSION(4:8),' 0.10')
                        if abs(HDR.VERSION - 0.10) < 2*eps
                                HDR.Dur = fread(HDR.FILE.FID,1,'float64');	% 8 Byte # duration of data record in sec
                        else
                                tmp  = fread(HDR.FILE.FID,2,'uint32');  % 8 Byte # duration of data record in sec
                                tmp1 = warning('off');
                                HDR.Dur = tmp(1)./tmp(2);
                                warning(tmp1);
                        end;
                        HDR.NS =   fread(HDR.FILE.FID,1,'uint32');     % 4 Byte # of signals
                else 
                        H1(193:256)= fread(HDR.FILE.FID,[1,256-192],'uchar');     %
                        H1 = char(H1); 
                        HDR.PID = deblank(char(H1(9:88)));                  % 80 Byte local patient identification
                        HDR.RID = deblank(char(H1(89:168)));                % 80 Byte local recording identification
			[HDR.Patient.Id,tmp] = strtok(HDR.PID,' ');
			[tmp1,tmp] = strtok(tmp,' ');
			[tmp1,tmp] = strtok(tmp,' ');
                        HDR.Patient.Name = tmp(2:end); 
                                
                        tmp=(find((H1<32) | (H1>126))); 		%%% syntax for Matlab
                        if ~isempty(tmp) %%%%% not EDF because filled out with ASCII(0) - should be spaces
                                %H1(tmp)=32; 
                                HDR.ErrNo=[1025,HDR.ErrNo];
                        end;
                        
                        tmp = repmat(' ',1,22);
                        tmp([3:4,6:7,9:10,12:13,15:16,18:19]) = H1(168+[7:8,4:5,1:2,9:10,12:13,15:16]);
                        tmp1 = str2double(tmp);
                        if length(tmp1)==6,
                                HDR.T0(1:6) = tmp1;
                        end;
                        
                        if any(isnan(HDR.T0)),
                                HDR.ErrNo = [1032,HDR.ErrNo];
                                
                                tmp = H1(168 + [1:16]);
                                tmp(tmp=='.' | tmp==':' | tmp=='/' | tmp=='-') = ' ';
                                tmp1 = str2double(tmp(1:8));
                                if length(tmp1)==3,
                                        HDR.T0 = tmp1([3,2,1]);
                                end;	
                                tmp1 = str2double(tmp(9:16));
                                if length(tmp1)==3,
                                        HDR.T0(4:6) = tmp1; 
                                end;
                                if any(isnan(HDR.T0)),
                                        HDR.ErrNo = [2,HDR.ErrNo];
                                end;
                        end;
                        
                        % Y2K compatibility until year 2084
                        if HDR.T0(1) < 85    % for biomedical data recorded in the 1950's and converted to EDF
                                HDR.T0(1) = 2000+HDR.T0(1);
                        elseif HDR.T0(1) < 100
                                HDR.T0(1) = 1900+HDR.T0(1);
                                %else % already corrected, do not change
                        end;
                        
                        HDR.HeadLen = str2double(H1(185:192));           % 8 Bytes  Length of Header
                        HDR.reserved1=H1(193:236);              % 44 Bytes reserved   
                        HDR.NRec    = str2double(H1(237:244));     % 8 Bytes  # of data records
                        HDR.Dur     = str2double(H1(245:252));     % 8 Bytes  # duration of data record in sec
                        HDR.NS      = str2double(H1(253:256));     % 4 Bytes  # of signals
                        HDR.AS.H1 = H1;	                     % for debugging the EDF Header
                        
                        if strcmp(HDR.reserved1(1:4),'EDF+'),	% EDF+ specific header information 
                                [HDR.Patient.Id,   tmp] = strtok(HDR.PID,' ');
                                [sex, tmp] = strtok(tmp,' ');
                                [bd, tmp] = strtok(tmp,' ');
                                [HDR.Patient.Name, tmp] = strtok(tmp,' ');
                                HDR.Patient.Sex = any(sex(1)=='mM') + any(sex(1)=='Ff')*2;
                                HDR.Patient.Birthday = datevec(bd);
                                
                                [chk, tmp] = strtok(HDR.RID,' ');
                                if ~strcmp(chk,'Startdate')
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN: EDF+ header is corrupted.\n');
                                end;
                                [HDR.Date2, tmp] = strtok(tmp,' ');
                                [HDR.ID.Investigation, tmp] = strtok(tmp,' ');
                                [HDR.ID.Investigator,  tmp] = strtok(tmp,' ');
                                [HDR.ID.Equiment, tmp] = strtok(tmp,' ');
                        end;
                end;
                
                if any(size(HDR.NS)~=1) %%%%% not EDF because filled out with ASCII(0) - should be spaces
                        fprintf(HDR.FILE.stderr, 'Warning SOPEN (GDF/EDF/BDF): invalid NS-value in header of %s\n',HDR.FileName);
                        HDR.ErrNo=[1040,HDR.ErrNo];
                        HDR.NS=1;
                end;
                % Octave assumes HDR.NS is a matrix instead of a scalare. Therefore, we need
                % Otherwise, eye(HDR.NS) will be executed as eye(size(HDR.NS)).
                HDR.NS = HDR.NS(1);     
                
                if isempty(HDR.HeadLen) %%%%% not EDF because filled out with ASCII(0) - should be spaces
                        HDR.ErrNo=[1056,HDR.ErrNo];
                        HDR.HeadLen=256*(1+HDR.NS);
                end;
                
                if isempty(HDR.NRec) %%%%% not EDF because filled out with ASCII(0) - should be spaces
                        HDR.ErrNo=[1027,HDR.ErrNo];
                        HDR.NRec = -1;
                end;
                
                if isempty(HDR.Dur) %%%%% not EDF because filled out with ASCII(0) - should be spaces
                        HDR.ErrNo=[1088,HDR.ErrNo];
                        HDR.Dur=30;
                end;
                
                if  any(HDR.T0>[2084 12 31 24 59 59]) | any(HDR.T0<[1985 1 1 0 0 0])
                        HDR.ErrNo = [4, HDR.ErrNo];
                end;
                
                %%% Read variable Header %%%
                %if ~strcmp(HDR.VERSION(1:3),'GDF'),
                if ~strcmp(HDR.TYPE,'GDF'),
                        idx1=cumsum([0 H2idx]);
                        idx2=HDR.NS*idx1;
                        
                        h2=zeros(HDR.NS,256);
                        [H2,count]=fread(HDR.FILE.FID,HDR.NS*256,'uchar');
                        if count < HDR.NS*256 
                                HDR.ErrNo=[8,HDR.ErrNo];
                                return; 
                        end;
                        
                        %tmp=find((H2<32) | (H2>126)); % would confirm 
                        tmp = find((H2<32) | ((H2>126) & (H2~=255) & (H2~=181)& (H2~=230))); 
                        if ~isempty(tmp) %%%%% not EDF because filled out with ASCII(0) - should be spaces
                                H2(tmp) = 32; 
                                HDR.ErrNo = [1026,HDR.ErrNo];
                        end;
                        
                        for k=1:length(H2idx);
                                %disp([k size(H2) idx2(k) idx2(k+1) H2idx(k)]);
                                h2(:,idx1(k)+1:idx1(k+1))=reshape(H2(idx2(k)+1:idx2(k+1)),H2idx(k),HDR.NS)';
                        end;
                        h2=char(h2);
                        
                        HDR.Label      =            h2(:,idx1(1)+1:idx1(2));
                        HDR.Transducer =            h2(:,idx1(2)+1:idx1(3));
                        HDR.PhysDim    =            h2(:,idx1(3)+1:idx1(4));
                        HDR.PhysMin    = str2double(h2(:,idx1(4)+1:idx1(5)));
                        HDR.PhysMax    = str2double(h2(:,idx1(5)+1:idx1(6)));
                        HDR.DigMin     = str2double(h2(:,idx1(6)+1:idx1(7)));
                        HDR.DigMax     = str2double(h2(:,idx1(7)+1:idx1(8)));
                        HDR.PreFilt    =            h2(:,idx1(8)+1:idx1(9));
                        HDR.AS.SPR        = str2double(h2(:,idx1(9)+1:idx1(10)));
                        %if ~all(abs(HDR.VERSION)==[255,abs('BIOSEMI')]),
                        if (HDR.VERSION ~= -1),
                                HDR.GDFTYP     = 3*ones(1,HDR.NS);	%	datatype
                        else
                                HDR.GDFTYP     = (255+24)*ones(1,HDR.NS);	%	datatype
                        end;
                        
                        if isempty(HDR.AS.SPR), 
                                fprintf(HDR.FILE.stderr, 'Warning SOPEN (GDF/EDF/BDF): invalid SPR-value in header of %s\n',HDR.FileName);
                                HDR.AS.SPR=ones(HDR.NS,1);
                                HDR.ErrNo=[1028,HDR.ErrNo];
                        end;
                else
                        if (ftell(HDR.FILE.FID)~=256),
                                error('position error');
                        end;	 
                        HDR.Label      =  setstr(fread(HDR.FILE.FID,[16,HDR.NS],'uchar')');		
                        HDR.Transducer =  setstr(fread(HDR.FILE.FID,[80,HDR.NS],'uchar')');	
                        HDR.PhysDim    =  setstr(fread(HDR.FILE.FID,[ 8,HDR.NS],'uchar')');
                        HDR.PhysMin    =         fread(HDR.FILE.FID,[HDR.NS,1],'float64');	
                        HDR.PhysMax    =         fread(HDR.FILE.FID,[HDR.NS,1],'float64');	
                        
                        %HDR.DigMin     =         fread(HDR.FILE.FID,[HDR.NS,1],'int64');	
                        %HDR.DigMax     =         fread(HDR.FILE.FID,[HDR.NS,1],'int64');	
                        tmp            =         fread(HDR.FILE.FID,[2*HDR.NS,1],'int32');
                        HDR.DigMin     = tmp((1:HDR.NS)*2-1);
                        tmp            =         fread(HDR.FILE.FID,[2*HDR.NS,1],'int32');	
                        HDR.DigMax     = tmp((1:HDR.NS)*2-1);
                        HDR.THRESHOLD  = [HDR.DigMin,HDR.DigMax];       % automated overflow detection 
                        
			if (HDR.VERSION < 1.9),
                                HDR.PreFilt    = char(fread(HDR.FILE.FID,[80,HDR.NS],'uchar')');	%	
                                HDR.AS.SPR     =       fread(HDR.FILE.FID,[ 1,HDR.NS],'uint32')';	%	samples per data record
                                HDR.GDFTYP     =       fread(HDR.FILE.FID,[ 1,HDR.NS],'uint32');	%	datatype
                        else
                                HDR.PreFilt    = char(fread(HDR.FILE.FID,[80-12,HDR.NS],'uchar')');	%	
                                HDR.Filter.LowPass  =  fread(HDR.FILE.FID,[ 1,HDR.NS],'float32');	% 
                                HDR.Filter.HighPass =  fread(HDR.FILE.FID,[ 1,HDR.NS],'float32');	%
                                HDR.Filter.Notch    =  fread(HDR.FILE.FID,[ 1,HDR.NS],'float32');	%
                                HDR.AS.SPR     =       fread(HDR.FILE.FID,[ 1,HDR.NS],'uint32')';	%	samples per data record
                                HDR.GDFTYP     =       fread(HDR.FILE.FID,[ 1,HDR.NS],'uint32');	%	datatype
                                HDR.ELEC.XYZ = fread(HDR.FILE.FID,[ 3,HDR.NS],'float32')';	%	datatype
                                tmp          = fread(HDR.FILE.FID,[HDR.NS, 1],'uint8');	%	datatype
                                HDR.REC.Impedance = 2.^(tmp/8);
			end;
                end;

                if HDR.VERSION<1.9,
                HDR.Filter.LowPass = repmat(nan,1,HDR.NS);
                HDR.Filter.HighPass = repmat(nan,1,HDR.NS);
                HDR.Filter.Notch = repmat(nan,1,HDR.NS);
                for k=1:HDR.NS,
                        tmp = HDR.PreFilt(k,:);
                        
                        ixh=strfind(tmp,'HP');
                        ixl=strfind(tmp,'LP');
                        ixn=strfind(tmp,'Notch');
                        ix =strfind(lower(tmp),'hz');
                        %tmp(tmp==':')=' ';
                        %try;
                                if any(tmp==';')
                                        [tok1,tmp] = strtok(tmp,';');
                                        [tok2,tmp] = strtok(tmp,';');
                                        [tok3,tmp] = strtok(tmp,';');
                                else
                                        [tok1,tmp] = strtok(tmp,' ');
                                        [tok2,tmp] = strtok(tmp,' ');
                                        [tok3,tmp] = strtok(tmp,' ');
                                end;
                                [T1, F1 ] = strtok(tok1,': ');
                                [T2, F2 ] = strtok(tok2,': ');
                                [T3, F3 ] = strtok(tok3,': ');
                                
                                [F1 ] = strtok(F1,': ');
                                [F2 ] = strtok(F2,': ');
                                [F3 ] = strtok(F3,': ');
                                
                                F1(find(F1==','))='.';
                                F2(find(F2==','))='.';
                                F3(find(F3==','))='.';
                                
                                if strcmp(F1,'DC'), F1='0'; end;
                                if strcmp(F2,'DC'), F2='0'; end;
                                if strcmp(F3,'DC'), F3='0'; end;
                                
                                tmp = strfind(lower(F1),'hz');
                                if ~isempty(tmp), F1=F1(1:tmp-1); end;
                                tmp = strfind(lower(F2),'hz');
                                if ~isempty(tmp), F2=F2(1:tmp-1); end;
                                tmp = strfind(lower(F3),'hz');
                                if ~isempty(tmp), F3=F3(1:tmp-1); end;
                                
                                if strcmp(T1,'LP'), 
                                        HDR.Filter.LowPass(k) =str2double(F1);
                                elseif strcmp(T1,'HP'), 
                                        HDR.Filter.HighPass(k)=str2double(F1);
                                elseif strcmp(T1,'Notch'), 
                                        HDR.Filter.Notch(k)   =str2double(F1);
                                end;
                                if strcmp(T2,'LP'), 
                                        HDR.Filter.LowPass(k) =str2double(F2);
                                elseif strcmp(T2,'HP'), 
                                        HDR.Filter.HighPass(k)=str2double(F2);
                                elseif strcmp(T2,'Notch'), 
                                        HDR.Filter.Notch(k)   =str2double(F2);
                                end;
                                if strcmp(T3,'LP'), 
                                        HDR.Filter.LowPass(k) =str2double(F3);
                                elseif strcmp(T3,'HP'), 
                                        HDR.Filter.HighPass(k)=str2double(F3);
                                elseif strcmp(T3,'Notch'), 
                                        HDR.Filter.Notch(k)   =str2double(F3);
                                end;
                                %catch
                        %        fprintf(2,'Cannot interpret: %s\n',HDR.PreFilt(k,:));
                        %end;
                end;
                end
        
                if any(HDR.PhysMax==HDR.PhysMin), HDR.ErrNo=[1029,HDR.ErrNo]; end;	
                if any(HDR.DigMax ==HDR.DigMin ), HDR.ErrNo=[1030,HDR.ErrNo]; end;	
                
                if HDR.NS & ~any(HDR.GDFTYP(1)==[0,16:18])
                        HDR.Cal = (HDR.PhysMax-HDR.PhysMin)./(HDR.DigMax-HDR.DigMin);
                        HDR.Off = HDR.PhysMin - HDR.Cal .* HDR.DigMin;
                else
                        HDR.Off = zeros(HDR.NS,1);
                        HDR.Cal = ones(HDR.NS,1);
                end;
                HDR.EDF.SampleRate = HDR.AS.SPR / HDR.Dur;
                HDR.SPR=1;
                for k=1:HDR.NS,
                        HDR.SPR = lcm(HDR.SPR,HDR.AS.SPR(k));
                end;
                HDR.SampleRate = HDR.SPR/HDR.Dur;
                
                HDR.AS.spb = sum(HDR.AS.SPR);	% Samples per Block
                HDR.AS.bi = [0;cumsum(HDR.AS.SPR(:))]; 
                HDR.AS.BPR  = ceil(HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)'); 
                while any(HDR.AS.BPR  ~= HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)');
                        fprintf(2,'\nError SOPEN (GDF/EDF/BDF): block configuration in file %s not supported.\n',HDR.FileName);
                end;
                HDR.AS.SAMECHANTYP = all(HDR.AS.BPR == (HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)')) & ~any(diff(HDR.GDFTYP)); 
                HDR.AS.bpb = sum(ceil(HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)'));	% Bytes per Block
                HDR.AS.EVENTTABLEPOS = -1;
                
                HDR.Calib = [HDR.Off'; diag(HDR.Cal)];
                HDR.AS.endpos = HDR.FILE.size;
                
                if HDR.NRec == -1   % unknown record size, determine correct NRec
                        HDR.NRec = floor((HDR.AS.endpos - HDR.HeadLen) / HDR.AS.bpb);
                elseif  (HDR.NRec*HDR.AS.bpb) ~= (HDR.AS.endpos - HDR.HeadLen);
                        %if ~strcmp(HDR.VERSION(1:3),'GDF'),
                        if ~strcmp(HDR.TYPE,'GDF'),
                                HDR.ErrNo= [16,HDR.ErrNo];
                                tmp = HDR.NRec; 
                                HDR.NRec = floor((HDR.AS.endpos - HDR.HeadLen) / HDR.AS.bpb);
                                fprintf(2,'\nWarning SOPEN (GDF/EDF/BDF): filesize (%i) of %s does not fit headerinformation (NRec = %i not %i)\n',HDR.AS.endpos,HDR.FileName,tmp,HDR.NRec);
                        else
                                HDR.AS.EVENTTABLEPOS = HDR.HeadLen + HDR.AS.bpb*HDR.NRec;
                        end;
                end; 
                
                % prepare SREAD for different data types 
                n = 0; 
                typ = [-1;HDR.GDFTYP(:)];
                for k = 1:HDR.NS; 
                        if (typ(k) == typ(k+1)),
                                HDR.AS.c(n)   = HDR.AS.c(n)  + HDR.AS.SPR(k);
                                HDR.AS.c2(n)  = HDR.AS.c2(n) + HDR.AS.BPR(k);
                        else
                                n = n + 1; 
                                HDR.AS.c(n)   = HDR.AS.SPR(k);
                                HDR.AS.c2(n)  = HDR.AS.BPR(k);
                                HDR.AS.TYP(n) = HDR.GDFTYP(k);
                        end;
                end;
                
                
                if 0, 
                        
                elseif strcmp(HDR.TYPE,'GDF') & (HDR.AS.EVENTTABLEPOS > 0),  
                        status = fseek(HDR.FILE.FID, HDR.AS.EVENTTABLEPOS, 'bof');
                        [EVENT.Version,c] = fread(HDR.FILE.FID,1,'char');
                        HDR.EVENT.SampleRate = [1,256,65536]*fread(HDR.FILE.FID,3,'uint8');
                        if ~HDR.EVENT.SampleRate, % ... is not defined in GDF 1.24 or earlier
                                HDR.EVENT.SampleRate = HDR.SampleRate; 
                        end;
                        [EVENT.N,c] = fread(HDR.FILE.FID,1,'uint32');
                        [HDR.EVENT.POS,c1] = fread(HDR.FILE.FID,[EVENT.N,1],'uint32');
                        [HDR.EVENT.TYP,c2] = fread(HDR.FILE.FID,[EVENT.N,1],'uint16');
                        if EVENT.Version==1,
                                if any([c1,c2]~=EVENT.N) | (HDR.AS.endpos~=HDR.AS.EVENTTABLEPOS+8+EVENT.N*6),
                                        fprintf(2,'\nERROR SOPEN (GDF/EDF/BDF): Eventtable corrupted in file %s\n',HDR.FileName);
                                end
                                
                        elseif EVENT.Version==3,
                                [HDR.EVENT.CHN,c3] = fread(HDR.FILE.FID,[EVENT.N,1],'uint16');
                                [HDR.EVENT.DUR,c4] = fread(HDR.FILE.FID,[EVENT.N,1],'uint32');
                                if any([c1,c2,c3,c4]~=EVENT.N) | (HDR.AS.endpos~=HDR.AS.EVENTTABLEPOS+8+EVENT.N*12),
                                        fprintf(2,'\nERROR SOPEN (GDF/EDF/BDF): Eventtable corrupted in file %s\n',HDR.FileName);
                                end
                                
                        else
                                fprintf(2,'\nWarning SOPEN (GDF/EDF/BDF): File %s corrupted (Eventtable version %i ).\n',HDR.FileName,EVENT.Version);
                        end;
                        HDR.AS.endpos = HDR.AS.EVENTTABLEPOS;   % set end of data block, might be important for SSEEK
                        
                        % Classlabels according to 
                        % http://cvs.sourceforge.net/viewcvs.py/*checkout*/biosig/biosig/t200/eventcodes.txt
                        if (length(HDR.EVENT.TYP)>0)
                                ix = (HDR.EVENT.TYP>hex2dec('0300')) & (HDR.EVENT.TYP<hex2dec('030d'));
                                ix = ix | ((HDR.EVENT.TYP>=hex2dec('0320')) & (HDR.EVENT.TYP<=hex2dec('037f')));
                                ix = ix | (HDR.EVENT.TYP==hex2dec('030f')); % unknown/undefined cue
                                HDR.Classlabel = mod(HDR.EVENT.TYP(ix),256);
                                HDR.Classlabel(HDR.Classlabel==15) = NaN; % unknown/undefined cue
                        end;
                        
                        % Trigger information and Artifact Selection 
                        ix = find(HDR.EVENT.TYP==hex2dec('0300')); 
                        HDR.TRIG = HDR.EVENT.POS(ix);
                        ArtifactSelection = repmat(logical(0),length(ix),1);
                        for k = 1:length(ix),
                                ix2 = find(HDR.EVENT.POS(ix(k))==HDR.EVENT.POS);
                                if any(HDR.EVENT.TYP(ix2)==hex2dec('03ff'))
                                        ArtifactSelection(k) = logical(1);                
                                end;
                        end;
                        if any(ArtifactSelection), % define only if necessary
                                HDR.ArtifactSelection = ArtifactSelection; 
                        end;
                        
                elseif strcmp(HDR.TYPE,'EDF') & (length(strmatch('EDF Annotations',HDR.Label))==1),
                        % EDF+: 
                        tmp = strmatch('EDF Annotations',HDR.Label);
                        HDR.EDF.Annotations = tmp;
                        %       HDR.Cal(HDR.EDF.Annotations) = 1;
                        %       HDR.Off(HDR.EDF.Annotations) = 0;
                        %	HDR.Calib(:,tmp) = [];
                        
                        status = fseek(HDR.FILE.FID,HDR.HeadLen+HDR.AS.bi(HDR.EDF.Annotations)*2,'bof');
                        t = fread(HDR.FILE.FID,inf,[int2str(HDR.AS.SPR(HDR.EDF.Annotations)*2),'*uchar=>uchar'],HDR.AS.bpb-HDR.AS.SPR(HDR.EDF.Annotations)*2);
                        HDR.EDF.ANNONS = t';
                        lt = length(t);
                        EVENTTABLE = repmat(0,lt/2,4);
                        TeegType = repmat({''},lt/2,1);
                        TeegDesc = TeegType; 
                        N = 0; 
                        ix = 1; 
                        t = [t;' ']';
                        FLAG.DUR = 0; 
                        while ix < length(t),
                                while (ix<=lt) & (t(ix)==0), ix = ix+1; end;
                                ix1 = ix; 
                                while (ix<=lt) & (t(ix)~=0), ix = ix+1; end;
                                ix2 = ix; 
                                if (ix < lt),
                                        v = t(ix1:ix2-1);
                                        [s1,v]=strtok(v,20);
                                        s1(s1==21) = 0;
                                        tmp=str2double(char(s1));
                                        
                                        [s2,v]=strtok(v,20);
                                        [s3,v]=strtok(v,20);
                                        
                                        N = N+1;
                                        EVENTTABLE(N,2) = tmp(1);
                                        if length(tmp)>1,
                                                EVENTTABLE(N,3) = tmp(2);
                                                FLAG.DUR = 1; 
                                        end;
                                        TeegType{N} = char(s2);
                                        TeegDesc{N} = char(s3);
                                end;
                        end;
                        HDR.EVENT.TYP = ones(N,1);
                        HDR.EVENT.POS = EVENTTABLE(1:N,2)*HDR.SampleRate;
                        if FLAG.DUR, 
                                HDR.EVENT.DUR = EVENTTABLE(1:N,3)*HDR.SampleRate;
                                HDR.EVENT.CHN = zeros(N,1);
                        end;
                        [HDR.EVENT.CodeDesc,HDR.EVENT.CodeIndex,TYP] = unique(TeegType(1:N));
                        if length(HDR.EVENT.CodeDesc)<16;
                                HDR.EVENT.TYP = TYP;
                        end;
                end;
                
                status = fseek(HDR.FILE.FID, HDR.HeadLen, 'bof');
                HDR.FILE.POS = 0;
                HDR.FILE.OPEN = 1;
                
                %%% Event file stored in GDF-format
                if ~any([HDR.NS,HDR.NRec,~length(HDR.EVENT.POS)]);
                        HDR.TYPE = 'EVENT';
                        HDR = sclose(HDR);
                end;	
                
        elseif any(PERMISSION=='w');                %%%%%%% ============= WRITE ===========%%%%%%%%%%%%        
                if strcmp(HDR.TYPE,'EDF')
                        HDR.VERSION = 0;
                elseif strcmp(HDR.TYPE,'GDF') 
                        HDR.VERSION = 1.25;     %% stable version 
                        %HDR.VERSION = 1.91;     %% testing 
                elseif strcmp(HDR.TYPE,'BDF'),
                        HDR.VERSION = -1;
                end;

                if ~isfield(HDR,'RID')
                        HDR.RID=setstr(32+zeros(1,80));
                end;
                if ~isfield(HDR,'T0')
                        HDR.T0=zeros(1,6);
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.T0 not defined\n');
                elseif any(isnan(HDR.T0))
                        HDR.T0(isnan(HDR.T0))=0;
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.T0 not completely defined\n');
                end;
                if ~isfield(HDR,'Patient')
			HDR.Patient.Sex = 0; 
			HDR.Patient.Handedness = 0; 
			HDR.Patient.Birthday = zeros(1,6);
			HDR.Patient.Headsize = [NaN,NaN,NaN]; 
                        HDR.Patient.Weight = 0;
                        HDR.Patient.Height = 0;
                end;
                if ~isfield(HDR.Patient,'Name')
			HDR.Patient.Name = 'X'; 
                end;
                if ~isfield(HDR.Patient,'Id')
			HDR.Patient.Id = 'X'; 
                end;
                if ~isfield(HDR.Patient,'Sex')
			HDR.Patient.Sex = 0; 
		elseif strcmpi(HDR.Patient.Sex,'m') | strcmpi(HDR.Patient.Sex,'male')
			HDR.Patient.Sex = 1; 
		elseif strcmpi(HDR.Patient.Sex,'f') | strcmpi(HDR.Patient.Sex,'female')
			HDR.Patient.Sex = 2; 
		else	
			HDR.Patient.Sex = 0; 
                end;
                
                if ~isfield(HDR.Patient,'Handedness')
			HDR.Patient.Handedness = 0; 
		elseif strcmpi(HDR.Patient.Handedness,'r') | strcmpi(HDR.Patient.Handedness,'right')
			HDR.Patient.Handedness = 1; 
		elseif strcmpi(HDR.Patient.Handedness,'l') | strcmpi(HDR.Patient.Handedness,'left')
			HDR.Patient.Handedness = 2; 
		else	
			HDR.Patient.Handedness = 0; 
                end;
                if ~isfield(HDR.Patient,'Impairment.Visual')
                        HDR.Patient.Impairment.Visual = 0;
                elseif strcmpi(HDR.Patient.Impairment.Visual,'NO') | strcmpi(HDR.Patient.Impairment.Visual,'NO')
                        HDR.Patient.Impairment.Visual = 1;
                elseif strcmpi(HDR.Patient.Impairment.Visual,'Y') | strcmpi(HDR.Patient.Impairment.Visual,'YES')
                        HDR.Patient.Impairment.Visual = 2;
                elseif strncmpi(HDR.Patient.Impairment.Visual,'corr',4)
                        HDR.Patient.Impairment.Visual = 3;
                elseif isnumeric(HDR.Patient.Impairment.Visual)
                else 
                        HDR.Patient.Impairment.Visual = 0;
                end;
                if ~isfield(HDR.Patient,'Smoking')
                        HDR.Patient.Smoking = 0;
                elseif strcmpi(HDR.Patient.Smoking,'NO') | strcmpi(HDR.Patient.Smoking,'NO')
                        HDR.Patient.Smoking = 1;
                elseif strcmpi(HDR.Patient.Smoking,'Y') | strcmpi(HDR.Patient.Smoking,'YES')
                        HDR.Patient.Smoking = 2;
                elseif isnumeric(HDR.Patient.Smoking)
                else 
                        HDR.Patient.Smoking = 0;
                end;
                if ~isfield(HDR.Patient,'AlcoholAbuse')
                        HDR.Patient.AlcoholAbuse = 0;
                elseif strcmpi(HDR.Patient.AlcoholAbuse,'NO') | strcmpi(HDR.Patient.AlcoholAbuse,'NO')
                        HDR.Patient.AlcoholAbuse = 1;
                elseif strcmpi(HDR.Patient.AlcoholAbuse,'Y') | strcmpi(HDR.Patient.AlcoholAbuse,'YES')
                        HDR.Patient.AlcoholAbuse = 2;
                elseif isnumeric(HDR.Patient.AlcoholAbuse)
                else 
                        HDR.Patient.AlcoholAbuse = 0;
                end;
                if ~isfield(HDR.Patient,'DrugAbuse')
                        HDR.Patient.DrugAbuse = 0;
                elseif strcmpi(HDR.Patient.DrugAbuse,'NO') | strcmpi(HDR.Patient.DrugAbuse,'NO')
                        HDR.Patient.DrugAbuse = 1;
                elseif strcmpi(HDR.Patient.DrugAbuse,'Y') | strcmpi(HDR.Patient.DrugAbuse,'YES')
                        HDR.Patient.DrugAbuse = 2;
                elseif isnumeric(HDR.Patient.DrugAbuse)
                else 
                        HDR.Patient.DrugAbuse = 0;
                end;
                if ~isfield(HDR.Patient,'Medication')
                        HDR.Patient.Medication = 0;
                elseif strcmpi(HDR.Patient.Medication,'NO') | strcmpi(HDR.Patient.Medication,'NO')
                        HDR.Patient.Medication = 1;
                elseif strcmpi(HDR.Patient.Medication,'Y') | strcmpi(HDR.Patient.Medication,'YES')
                        HDR.Patient.Medication = 2;
                else 
                        HDR.Patient.Medication = 0;
                end;
                if ~isfield(HDR.Patient,'Weight')
                        HDR.Patient.Weight = 0; 
                elseif (HDR.Patient.Weight > 254),
			HDR.Patient.Weight = 255; 
                elseif isnan(HDR.Patient.Weight) | (isnan(HDR.Patient.Weight)<0)
			HDR.Patient.Weight = 0; 
                end;
                if ~isfield(HDR.Patient,'Height')
			HDR.Patient.Height = 0; 
                elseif (HDR.Patient.Height > 254),
			HDR.Patient.Height = 255; 
                elseif isnan(HDR.Patient.Height) | (isnan(HDR.Patient.Height)<0)
			HDR.Patient.Height = 0; 
                end;
                if ~isfield(HDR.Patient,'Birthday') 
			if ~isfield(HDR.Patient,'Age')
				HDR.Patient.Birthday = zeros(1,6);
                        elseif isnan(HDR.Patient.Age) 
				HDR.Patient.Birthday = zeros(1,6);
			else
				HDR.Patient.Birthday = datevec(datenum(HDR.T0) + HDR.Patient.Age*365.25);
			end;	
                end;
                if ~isfield(HDR.Patient,'Headsize')
			HDR.Patient.Headsize = [NaN,NaN,NaN]; 
		elseif ~isnumeric(HDR.Patient.Headsize)
                        fprintf('Warning SOPEN (GDF)-W: HDR.Patient.Headsize must be numeric.\n');
		elseif (numel(HDR.Patient.Headsize)~=3)
			tmp = [HDR.Patient.Headsize(:);NaN;NaN;NaN]';
			HDR.Patient.Headsize = HDR.Patient.Headsize(1:3); 
                end;
                if ~isfield(HDR,'REC')
			HDR.REC.LOC.RFC1876 = uint32([hex2dec('00292929'),48*36e5+2^31,15*36e5+2^31,35000]);
		end
		if ~isfield(HDR.REC.LOC,'RFC1876')	
			tmp = HDR.REC.LOC;
			HDR.REC.LOC.RFC1876 = [hex2dec('00292929'),tmp.Latitude*36e5,tmp.Longitude*36e5,tmp.Altitude*100];
		end
                HDR.ID.Equipment = [1,abs('BIOSIG ')];
                if ~isfield(HDR.ID,'Lab')
                    HDR.ID.Lab = repmat(32,1,8);
                end;
                if ~isfield(HDR.REC,'IPaddr')
                    HDR.REC.IPaddr = uint8(zeros(1,6));
                end;
                if ~isfield(HDR.ID,'Technician')
                    HDR.ID.Technician = repmat(32,1,8);
                end;
                if ~isfield(HDR,'NRec')
                        HDR.NRec=-1;
                end;
                if ~isfield(HDR,'Dur')
                        if HDR.NS>0,
                                fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.Dur not defined\n');
                        end;
                        HDR.Dur=NaN;
                end;
                if ~isfield(HDR,'NS')
                        HDR.ERROR = sprintf('Error SOPEN (GDF/EDF/BDF)-W: HDR.NS not defined\n');
                        HDR.ErrNo = HDR.ErrNo + 128;
                        return;
                end;
                if ~isfield(HDR,'SampleRate')
                        HDR.SampleRate = NaN;
                end;
                if ~isfield(HDR,'AS')
                        HDR.AS.SPR = NaN;
                end;
                if ~isfield(HDR.AS,'SPR')
                        HDR.AS.SPR = NaN;
                end;
                if ~isfield(HDR,'EDF')
                        HDR.EDF.SampleRate = repmat(HDR.SampleRate,HDR.NS,1);
                elseif ~isfield(HDR.EDF,'SampleRate')
                        HDR.EDF.SampleRate = repmat(HDR.SampleRate,HDR.NS,1);
                end;
                
                if ~HDR.NS,
                elseif ~isnan(HDR.Dur) & any(isnan(HDR.AS.SPR)) & ~any(isnan(HDR.EDF.SampleRate))
                        HDR.AS.SPR = HDR.EDF.SampleRate * HDR.Dur;
                elseif ~isnan(HDR.Dur) & ~any(isnan(HDR.AS.SPR)) & any(isnan(HDR.EDF.SampleRate))
                        HDR.SampleRate = HDR.Dur * HDR.AS.SPR;
                elseif isnan(HDR.Dur) & ~any(isnan(HDR.AS.SPR)) & ~any(isnan(HDR.EDF.SampleRate))
                        HDR.Dur = HDR.AS.SPR ./ HDR.SampleRate;
                        if all(HDR.Dur(1)==HDR.Dur)
                                HDR.Dur = HDR.Dur(1);
                        else
                                fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF): SPR and SampleRate do not fit\n');
                                [HDR.AS.SPR,HDR.SampleRate,HDR.Dur]
                        end;
                elseif ~isnan(HDR.Dur) & ~any(isnan(HDR.AS.SPR)) & ~any(isnan(HDR.EDF.SampleRate))
                        %% thats ok, 
                else
                        fprintf(HDR.FILE.stderr,'ERROR SOPEN (GDF/EDF/BDF): more than 1 of HDR.Dur, HDR.SampleRate, HDR.AS.SPR undefined.\n');
                        return; 
                end;
		
                %if (abs(HDR.VERSION(1))==255)  & strcmp(HDR.VERSION(2:8),'BIOSEMI'),
                if (HDR.VERSION == -1),
                        HDR.GDFTYP=255+24+zeros(1,HDR.NS);                        
                %elseif strcmp(HDR.VERSION,'0       '),
                elseif HDR.VERSION == 0,
                        HDR.GDFTYP=3+zeros(1,HDR.NS);                        
                %elseif strcmp(HDR.VERSION(1:3),'GDF'),
                elseif (HDR.VERSION>0),
                        if HDR.NS == 0;
                                HDR.GDFTYP = 3;
                        elseif ~isfield(HDR,'GDFTYP'),
                                HDR.ERROR = sprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.GDFTYP not defined\n');
                                HDR.ErrNo = HDR.ErrNo + 128;
                                % fclose(HDR.FILE.FID); return;
                        elseif length(HDR.GDFTYP)==1,
                                HDR.GDFTYP = HDR.GDFTYP(ones(HDR.NS,1));
                        elseif length(HDR.GDFTYP)>=HDR.NS,
                                HDR.GDFTYP = HDR.GDFTYP(1:HDR.NS);
                        end;
                else
                        fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF): invalid VERSION %s\n ',HDR.VERSION);
                        return;
                end;
                [tmp,HDR.THRESHOLD]=gdfdatatype(HDR.GDFTYP);
                
                if HDR.NS>0,	% header 2
                        % Check all fields of Header2
                        if ~isfield(HDR,'Label')
                                HDR.Label=setstr(32+zeros(HDR.NS,16));
                                if HDR.NS>0,
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.Label not defined\n');
                                end;
                        else
                                tmp = min(16,size(HDR.Label,2));
                                HDR.Label = [HDR.Label(1:HDR.NS,1:tmp), char(32+zeros(HDR.NS,16-tmp))];
                        end;
                        if ~isfield(HDR,'Transducer')
                                HDR.Transducer=setstr(32+zeros(HDR.NS,80));
                        else
                                tmp=min(80,size(HDR.Transducer,2));
                                HDR.Transducer=[HDR.Transducer(1:HDR.NS,1:tmp), setstr(32+zeros(HDR.NS,80-tmp))];
                        end;
                        if ~isfield(HDR,'Filter')
                                HDR.Filter.LowPass = repmat(NaN,1,HDR.NS); 
                                HDR.Filter.HighPass = repmat(NaN,1,HDR.NS); 
                                HDR.Filter.Notch = repmat(NaN,1,HDR.NS); 
                        else 
                                if ~isfield(HDR.Filter,'LowPass')
                                        HDR.Filter.LowPass = repmat(NaN,1,HDR.NS); 
                                elseif (numel(HDR.Filter.LowPass)==1)
                                        HDR.Filter.LowPass = repmat(HDR.Filter.LowPass,1,HDR.NS); 
                                elseif (numel(HDR.Filter.LowPass)~=HDR.NS)
                                        fprintf(HDR.FILE.stderr,'SOPEN (GDF) WRITE: HDR.Filter.LowPass has incorrrect number of field!\n')
                                end;
                                if ~isfield(HDR.Filter,'HighPass')
                                        HDR.Filter.HighPass = repmat(NaN,1,HDR.NS); 
                                elseif (numel(HDR.Filter.HighPass)==1)
                                        HDR.Filter.HighPass = repmat(HDR.Filter.HighPass,1,HDR.NS); 
                                elseif (numel(HDR.Filter.HighPass)~=HDR.NS)
                                        fprintf(HDR.FILE.stderr,'SOPEN (GDF) WRITE: HDR.Filter.HighPass has incorrrect number of field!\n')
                                end;
                                if ~isfield(HDR.Filter,'Notch')
                                        HDR.Filter.Notch = repmat(NaN,1,HDR.NS); 
                                elseif (numel(HDR.Filter.Notch)==1)
                                        HDR.Filter.Notch = repmat(HDR.Filter.Notch,1,HDR.NS); 
                                elseif (numel(HDR.Filter.Notch)~=HDR.NS)
                                        fprintf(HDR.FILE.stderr,'SOPEN (GDF) WRITE: HDR.Filter.Notch has incorrrect number of field!\n')
                                end;
                        end;
                        if ~isfield(HDR,'PreFilt')
                                HDR.PreFilt = setstr(32+zeros(HDR.NS,80));
                                if isfield(HDR,'Filter'),
                                        if isfield(HDR.Filter,'LowPass') & isfield(HDR.Filter,'HighPass') & isfield(HDR.Filter,'Notch'),
                                                if any(length(HDR.Filter.LowPass) == [1,HDR.NS]) & any(length(HDR.Filter.HighPass) == [1,HDR.NS]) & any(length(HDR.Filter.Notch) == [1,HDR.NS])
                                                        for k = 1:HDR.NS,
                                                                k1 = min(k,length(HDR.Filter.LowPass));
                                                                k2 = min(k,length(HDR.Filter.HighPass));
                                                                k3 = min(k,length(HDR.Filter.Notch));
                                                                PreFilt{k,1} = sprintf('LP: %5.f Hz; HP: %5.2f Hz; Notch: %i',HDR.Filter.LowPass(k1),HDR.Filter.HighPass(k2),HDR.Filter.Notch(k3));
                                                        end;
                                                        HDR.PreFilt = strvcat(PreFilt);
                                                end;
                                        end
                                end
                        elseif size(HDR.PreFilt,1)<HDR.NS,
                                HDR.PreFilt = repmat(HDR.PreFilt,HDR.NS,1);
                        end;
                        tmp = min(80,size(HDR.PreFilt,2));
                        HDR.PreFilt = [HDR.PreFilt(1:HDR.NS,1:tmp), setstr(32+zeros(HDR.NS,80-tmp))];
                        
                        if ~isfield(HDR,'PhysDim')
                                HDR.PhysDim=setstr(32+zeros(HDR.NS,8));
                                if HDR.NS>0,
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.PhysDim not defined\n');
                                end;
                        else
                                if size(HDR.PhysDim,1)==0,
                                        HDR.PhysDim = repmat(' ',HDR.NS,1);
                                elseif size(HDR.PhysDim,1)<HDR.NS,
                                        HDR.PhysDim = repmat(HDR.PhysDim,HDR.NS,1);
                                end;
                                tmp=min(8,size(HDR.PhysDim,2));
                                HDR.PhysDim=[HDR.PhysDim(1:HDR.NS,1:tmp), setstr(32+zeros(HDR.NS,8-tmp))];
                        end;
                        if any(HDR.GDFTYP(1)==[0,16:18])
                                HDR.DigMin  = repmat(-2^31,1,HDR.NS);
                                HDR.DigMax  = repmat( 2^31-1,1,HDR.NS);
                                HDR.PhysMin = repmat(-2^31,1,HDR.NS);
                                HDR.PhysMax = repmat( 2^31-1,1,HDR.NS);
                        end;
                        if ~isfield(HDR,'PhysMin')
                                if HDR.NS>0,
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.PhysMin not defined\n');
                                end
                                HDR.PhysMin=repmat(nan,HDR.NS,1);
                        else
                                HDR.PhysMin=HDR.PhysMin(1:HDR.NS);
                        end;
                        if ~isfield(HDR,'PhysMax')
                                if HDR.NS>0,
                                        fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.PhysMax not defined\n');
                                end;
                                HDR.PhysMax=repmat(nan,HDR.NS,1);
                        else
                                HDR.PhysMax=HDR.PhysMax(1:HDR.NS);
                        end;
                        if ~isfield(HDR,'DigMin')
                                if HDR.NS>0,
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF)-W: HDR.DigMin not defined\n');
                                end
                                HDR.DigMin=repmat(nan,HDR.NS,1);
                        else
                                HDR.DigMin=HDR.DigMin(1:HDR.NS);
                        end;
                        if ~isfield(HDR,'DigMax')
                                if HDR.NS>0,
                                        fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.DigMax not defined\n');
                                end;
                                HDR.DigMax=repmat(nan,HDR.NS,1);
                        else
                                HDR.DigMax=HDR.DigMax(1:HDR.NS);
                        end;
			flag = isfield(HDR,'ELEC');	
			if flag,
				flag = isfield(HDR.ELEC,'XYZ');
			end;		
			if ~flag,				
				HDR.ELEC.XYZ = repmat(NaN,HDR.NS,3); 
				HDR.ELEC.REF = repmat(NaN,1,3); 
				HDR.ELEC.GND = repmat(NaN,1,3); 
			elseif ~isnumeric(HDR.ELEC.XYZ)
                                fprintf('Warning SOPEN (GDF)-W: HDR.ELEC.LOC must be numeric.\n');
			elseif any(size(HDR.ELEC.XYZ)==[HDR.NS,3])
                                HDR.ELEC.REF = repmat(NaN,1,3); 
				HDR.ELEC.GND = repmat(NaN,1,3); 
			elseif any(size(HDR.ELEC.XYZ)==[HDR.NS+1,3])
                                HDR.ELEC.REF = HDR.ELEC.XYZ(HDR.NS+1,:); 
				HDR.ELEC.GND = repmat(NaN,1,3); 
			elseif any(size(HDR.ELEC.XYZ)==[HDR.NS+2,3])
                                HDR.ELEC.REF = HDR.ELEC.XYZ(HDR.NS+1,:); 
                                HDR.ELEC.GND = HDR.ELEC.XYZ(HDR.NS+2,:); 
			else
                                fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.ELEC.LOC not correctly defined\n');
				tmp = [HDR.ELEC.XYZ,repmat(NaN,size(HDR.ELEC.XYZ,1),3)];
				tmp = [tmp;repmat(NaN,HDR.NS+2,size(tmp,2))];
				HDR.ELEC.XYZ = tmp(1:HDR.NS,1:3);
                                HDR.ELEC.REF = HDR.ELEC.XYZ(HDR.NS+1,:); 
                                HDR.ELEC.GND = HDR.ELEC.XYZ(HDR.NS+2,:); 
    		        end;
	                if ~isfield(HDR.REC,'Impedance')
				HDR.REC.Impedance = repmat(NaN,HDR.NS,1); 
			elseif ~isnumeric(HDR.REC.Impedance)
                                fprintf('Warning SOPEN (GDF)-W: HDR.REC.Impedance must be numeric.\n');
			elseif (length(HDR.REC.Impedance)~=HDR.NS)
				sz = size(HDR.REC.Impedance(:));
				tmp = [HDR.REC.Impedance(:),repmat(NaN,sz(1),1);repmat(NaN,HDR.NS,sz(2)+1)];	
				HDR.REC.Impedance = tmp(1:HDR.NS,1); 
			end
                        
                        ix = find((HDR.DigMax(:)==HDR.DigMin(:)) & (HDR.PhysMax(:)==HDR.PhysMin(:)));
                        HDR.PhysMax(ix) = 1; 
                        HDR.PhysMin(ix) = 0; 
                        HDR.DigMax(ix) = 1; 
                        HDR.DigMin(ix) = 0; 
                        
                        if ~isfield(HDR.AS,'SPR')
                                if HDR.NS>0,
                                        fprintf('Warning SOPEN (GDF/EDF/BDF)-W: HDR.AS.SPR not defined\n');
                                end;
                                HDR.AS.SPR = repmat(nan,HDR.NS,1);
                                HDR.ERROR = sprintf('Error SOPEN (GDF/EDF/BDF)-W: HDR.AS.SPR not defined\n');
                                HDR.ErrNo = HDR.ErrNo + 128;
                                %fclose(HDR.FILE.FID); return;
                        else
                                HDR.AS.SPR=reshape(HDR.AS.SPR(1:HDR.NS),HDR.NS,1);
                        end;
                        
                end;	% header 2
                
                HDR.SPR = 1;
                for k=1:HDR.NS,
                        HDR.SPR = lcm(HDR.SPR,HDR.AS.SPR(k));
                end;
                
                
                %%%%%% generate Header 1, first 256 bytes 
                HDR.HeadLen=(HDR.NS+1)*256;
                %H1(1:8)=HDR.VERSION; %sprintf('%08i',HDR.VERSION);     % 8 Byte  Versionsnummer 
		sex = 'XMF';
		if ~HDR.Patient.Birthday(1), bd = 'X';
		else bd=datestr(HDR.Patient.Birthday,'dd-mmm-yyyy');
		end;
                if HDR.VERSION == -1,
                        H1 = [255,'BIOSEMI',repmat(32,1,248)];
			HDR.PID = [HDR.Patient.Id,' ',sex(HDR.Patient.Sex+1),' ',bd,' ',HDR.Patient.Name];
			HDR.RID = ['Startdate ',datestr(HDR.T0,'dd-mmm-yyyy')];
                elseif HDR.VERSION == 0,
			H1 = ['0       ',repmat(32,1,248)]; 
			HDR.PID = [HDR.Patient.Id,' ',sex(HDR.Patient.Sex+1),' ',bd,' ',HDR.Patient.Name];
			HDR.RID = ['Startdate ',datestr(HDR.T0,'dd-mmm-yyyy')];
                elseif HDR.VERSION > 0,
                        tmp = sprintf('%5.2f',HDR.VERSION);
                        H1 = ['GDF',tmp(1:5),repmat(32,1,248)];
			HDR.PID = [HDR.Patient.Id,' ',HDR.Patient.Name];
			% HDR.RID = 'Hospital_administration_Code Technician_ID [Equipment_ID]'
                else
                        fprintf(HDR.FILE.stderr,'Error SOPEN (GDF) WRITE: invalid version number %f\n',HDR.VERSION); 
                end;
		H1( 8+(1:length(HDR.PID))) = HDR.PID;
                H1(88+(1:length(HDR.RID))) = HDR.RID;
                %H1(185:192)=sprintf('%-8i',HDR.HeadLen);
                HDR.AS.spb = sum(HDR.AS.SPR);	% Samples per Block
                HDR.AS.bi  = [0;cumsum(HDR.AS.SPR)];
                HDR.AS.BPR = ceil(HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)');
                while HDR.NS & any(HDR.AS.BPR  ~= HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)');
                        fprintf(2,'\nWarning SOPEN (GDF/EDF/BDF): invalid block configuration in file %s.\n',HDR.FileName);
                        DIV = 2;
                        HDR.SPR    = HDR.SPR*DIV;
                        HDR.AS.SPR = HDR.AS.SPR*DIV;
                        HDR.Dur    = HDR.Dur*DIV; 
                        HDR.NRec   = HDR.NRec/DIV; 
                        HDR.AS.BPR = ceil(HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)');
                end;
                HDR.AS.SAMECHANTYP = all(HDR.AS.BPR == (HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)')) & ~any(diff(HDR.GDFTYP));
                HDR.AS.spb = sum(HDR.AS.SPR);	% Samples per Block
                HDR.AS.bi  = [0;cumsum(HDR.AS.SPR)];
                HDR.AS.bpb   = sum(ceil(HDR.AS.SPR(:).*GDFTYP_BYTE(HDR.GDFTYP(:)+1)'));	% Bytes per Block
                HDR.FILE.POS  = 0;
                
		if HDR.VERSION>=1.9,	% do some header checks
		if ~HDR.Patient.Sex,
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF) WRITE: HDR.Patient.Sex is not defined.\n'); 
		end;	
		if ~HDR.Patient.Handedness,
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF) WRITE: HDR.Patient.Handedness is not defined.\n'); 
		end;	
		if datenum([1900,1,1,0,0,0])>datenum(HDR.Patient.Birthday),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF) WRITE: HDR.Patient.Birthday is not correctly defined.\n'); 
		end;	
		if any(isnan(HDR.REC.Impedance(:))),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF) WRITE: HDR.REC.Impedance not correctly defined.\n'); 
		end;	
		if any(isnan(HDR.ELEC.XYZ(:))),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF) WRITE: HDR.ELEC.XYZ not correctly defined.\n'); 
		end;	
		elseif (HDR.VERSION == 0)
                        if sum(HDR.AS.bpb)>61440;
                                fprintf(HDR.FILE.stderr,'\nWarning SOPEN (EDF): One block exceeds 61440 bytes.\n')
                        end;
		end;
		                
                %%%%% Open File 
                if ~any(PERMISSION=='+') 
                        [HDR.FILE.FID,MESSAGE]=fopen(HDR.FileName,'w+b','ieee-le');          
                else  % (arg2=='w+')  % may be called only by SDFCLOSE
                        if HDR.FILE.OPEN==2 
                                [HDR.FILE.FID,MESSAGE]=fopen(HDR.FileName,'r+b','ieee-le');          
                        else
                                fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W+: Cannot open %s for write access\n',HDR.FileName);
                                return;
                        end;
                end;
                if HDR.FILE.FID<0 
                        %fprintf(HDR.FILE.stderr,'Error EDFOPEN: %s\n',MESSAGE);  
                        H1=MESSAGE;H2=[];
                        HDR.ErrNo = HDR.ErrNo + 32;
                        fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W: Could not open %s \n',HDR.FileName);
                        return;
                end;
                HDR.FILE.OPEN = 2;
                
                %if strcmp(HDR.VERSION(1:3),'GDF'),
                if (HDR.VERSION > 0),  % GDF
			if (HDR.VERSION >= 1.90)
                		H1(85) = mod(HDR.Patient.Medication,3)*64 + mod(HDR.Patient.DrugAbuse,3)*16 + mod(HDR.Patient.AlcoholAbuse,3)*4  + mod(HDR.Patient.Smoking,3);
                                H1(86) = HDR.Patient.Weight; 
                                H1(87) = HDR.Patient.Height; 
                		H1(88) = bitand(HDR.Patient.Sex,3) + bitand(HDR.Patient.Handedness,3)*4 + bitand(HDR.Patient.Impairment.Visual,3)*16;
                                if all(H1(153:156)==32)
                                        c = fwrite(HDR.FILE.FID,abs(H1(1:152)),'uchar');
                                        c = fwrite(HDR.FILE.FID,HDR.REC.LOC.RFC1876,'uint32');
                                else
                                        c = fwrite(HDR.FILE.FID,abs(H1(1:156)),'uchar');
                                        c = fwrite(HDR.FILE.FID,HDR.REC.LOC.RFC1876(2:4),'uint32');
                                end;
				tmp = [datenum(HDR.T0), datenum(HDR.Patient.Birthday)];
				tmp = floor([rem(tmp,1)*2^32;tmp]);
                                c = fwrite(HDR.FILE.FID,tmp,'uint32');
                                c=fwrite(HDR.FILE.FID,[HDR.HeadLen,0],'int32');
                                c=fwrite(HDR.FILE.FID,'b4om1.91',uint8'); % EP_ID=ones(8,1)*32;
				tmp = [HDR.REC.IPaddr, zeros(1,2)];
			        c=fwrite(HDR.FILE.FID,tmp(6:-1:1),'uint8'); % IP address
			        c=fwrite(HDR.FILE.FID,HDR.Patient.Headsize(1:3),'uint16'); % circumference, nasion-inion, left-right mastoid in [mm]
			        c=fwrite(HDR.FILE.FID,HDR.ELEC.REF(1:3),'float32'); % [X,Y,Z] position of reference electrode
			        c=fwrite(HDR.FILE.FID,HDR.ELEC.GND(1:3),'float32'); % [X,Y,Z] position of ground electrode
                        else
                                H1(169:184) = sprintf('%04i%02i%02i%02i%02i%02i%02i',floor(HDR.T0),floor(100*rem(HDR.T0(6),1)));
                                c=fwrite(HDR.FILE.FID,H1(1:184),'uchar');
                                c=fwrite(HDR.FILE.FID,[HDR.HeadLen,0],'int32');
                                c=fwrite(HDR.FILE.FID,HDR.ID.Equipment,'uint8'); % EP_ID=ones(8,1)*32;
                                c=fwrite(HDR.FILE.FID,HDR.ID.Lab,'uint8'); % Lab_ID=ones(8,1)*32;
                                c=fwrite(HDR.FILE.FID,HDR.ID.Technician,'uint8'); % T_ID=ones(8,1)*32;
                                c=fwrite(HDR.FILE.FID,ones(20,1)*32,'uint8'); % 
                        end;

                        %c=fwrite(HDR.FILE.FID,HDR.NRec,'int64');
                        c=fwrite(HDR.FILE.FID,[HDR.NRec,0],'int32');
                        %fwrite(HDR.FILE.FID,HDR.Dur,'float64');
                        [n,d]=rat(HDR.Dur); fwrite(HDR.FILE.FID,[n d], 'uint32');
                        c=fwrite(HDR.FILE.FID,HDR.NS,'uint32');
                else
                        H1(168+(1:16))=sprintf('%02i.%02i.%02i%02i:%02i:%02i',floor(rem(HDR.T0([3 2 1 4 5 6]),100)));
                        H1(185:192)=sprintf('%-8i',HDR.HeadLen);
                        H1(237:244)=sprintf('%-8i',HDR.NRec);
                        H1(245:252)=sprintf('%-8i',HDR.Dur);
                        H1(253:256)=sprintf('%-4i',HDR.NS);
                        H1(abs(H1)==0)=char(32); 
                        c=fwrite(HDR.FILE.FID,abs(H1),'uchar');
                end;

                %%%%%% generate Header 2,  NS*256 bytes 
                if HDR.NS>0, 
                        %if ~strcmp(HDR.VERSION(1:3),'GDF');
                        if ~(HDR.VERSION > 0);
                                sPhysMax=char(32+zeros(HDR.NS,8));
                                sPhysMin=char(32+zeros(HDR.NS,8));
                                for k=1:HDR.NS,
                                        tmp=sprintf('%-8g',HDR.PhysMin(k));
                                        lt=length(tmp);
                                        if lt<9
                                                sPhysMin(k,1:lt)=tmp;
                                        else
                                                if any(upper(tmp)=='E') | find(tmp=='.')>8,
                                                        fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W: PhysMin(%i) does not fit into header\n', k);
                                                else
                                                        sPhysMin(k,:)=tmp(1:8);
                                                end;
                                        end;
                                        tmp=sprintf('%-8g',HDR.PhysMax(k));
                                        lt=length(tmp);
                                        if lt<9
                                                sPhysMax(k,1:lt)=tmp;
                                        else
                                                if any(upper(tmp)=='E') | find(tmp=='.')>8,
                                                        fprintf(HDR.FILE.stderr,'Error SOPEN (GDF/EDF/BDF)-W: PhysMin(%i) does not fit into header\n', k);
                                                else
                                                        sPhysMax(k,:)=tmp(1:8);
                                                end;
                                        end;
                                end;
                                
                                idx1=cumsum([0 H2idx]);
                                idx2=HDR.NS*idx1;
                                h2=setstr(32*ones(HDR.NS,256));
                                size(h2);
                                h2(:,idx1(1)+1:idx1(2))=HDR.Label;
                                h2(:,idx1(2)+1:idx1(3))=HDR.Transducer;
                                h2(:,idx1(3)+1:idx1(4))=HDR.PhysDim;
                                %h2(:,idx1(4)+1:idx1(5))=sPhysMin;
                                %h2(:,idx1(5)+1:idx1(6))=sPhysMax;
                                h2(:,idx1(4)+1:idx1(5))=sPhysMin;
                                h2(:,idx1(5)+1:idx1(6))=sPhysMax;
                                h2(:,idx1(6)+1:idx1(7))=reshape(sprintf('%-8i',HDR.DigMin)',8,HDR.NS)';
                                h2(:,idx1(7)+1:idx1(8))=reshape(sprintf('%-8i',HDR.DigMax)',8,HDR.NS)';
                                h2(:,idx1(8)+1:idx1(9))=HDR.PreFilt;
                                h2(:,idx1(9)+1:idx1(10))=reshape(sprintf('%-8i',HDR.AS.SPR)',8,HDR.NS)';
                                h2(abs(h2)==0)=char(32);
                                for k=1:length(H2idx);
                                        fwrite(HDR.FILE.FID,abs(h2(:,idx1(k)+1:idx1(k+1)))','uchar');
                                end;
                        else
                                fwrite(HDR.FILE.FID, abs(HDR.Label)','uchar');
                                fwrite(HDR.FILE.FID, abs(HDR.Transducer)','uchar');
                                fwrite(HDR.FILE.FID, abs(HDR.PhysDim)','uchar');
                                fwrite(HDR.FILE.FID, HDR.PhysMin,'float64');
                                fwrite(HDR.FILE.FID, HDR.PhysMax,'float64');
                                if exist('OCTAVE_VERSION','builtin'),  % Octave does not support INT64 yet. 
                                        fwrite(HDR.FILE.FID, [HDR.DigMin(:),-(HDR.DigMin(:)<0)]','int32');
                                        fwrite(HDR.FILE.FID, [HDR.DigMax(:),-(HDR.DigMax(:)<0)]','int32');
                                else
                                        fwrite(HDR.FILE.FID, HDR.DigMin, 'int64');
                                        fwrite(HDR.FILE.FID, HDR.DigMax, 'int64');
                                end;
                                if (HDR.VERSION<1.9),
                                        fwrite(HDR.FILE.FID, abs(HDR.PreFilt)','uchar');
                                        fwrite(HDR.FILE.FID, HDR.AS.SPR,'uint32');
                                        fwrite(HDR.FILE.FID, HDR.GDFTYP,'uint32');
	                                fwrite(HDR.FILE.FID,32*ones(32,HDR.NS),'char');
                                else
                                        fwrite(HDR.FILE.FID, abs(HDR.PreFilt(:,1:68))','uchar');
                                        fwrite(HDR.FILE.FID, HDR.Filter.LowPass,'float32');
                                        fwrite(HDR.FILE.FID, HDR.Filter.HighPass,'float32');
                                        fwrite(HDR.FILE.FID, HDR.Filter.Notch,'float32');
                                        fwrite(HDR.FILE.FID, HDR.AS.SPR,'uint32');
                                        fwrite(HDR.FILE.FID, HDR.GDFTYP,'uint32');
                                        fwrite(HDR.FILE.FID, HDR.ELEC.XYZ','float32');
                                        fwrite(HDR.FILE.FID, max(0,min(255,round(log2(HDR.REC.Impedance)*8)')),'uint8');
                                        fwrite(HDR.FILE.FID,32*ones(32-13,HDR.NS),'char');
				end;
                        end;
                end;
                tmp = ftell(HDR.FILE.FID);
                if tmp ~= (256 * (HDR.NS+1)) 
                        fprintf(1,'Warning SOPEN (GDF/EDF/BDF)-WRITE: incorrect header length %i bytes\n',tmp);
                        %else   fprintf(1,'SOPEN (GDF/EDF/BDF) in write mode: header info stored correctly\n');
                end;        
                
        else % if arg2 is not 'r' or 'w'
                fprintf(HDR.FILE.stderr,'Warning SOPEN (GDF/EDF/BDF): Incorrect 2nd argument. \n');
        end;        
        
        if HDR.ErrNo>0
                fprintf(HDR.FILE.stderr,'ERROR %i SOPEN (GDF/EDF/BDF)\n',HDR.ErrNo);
        end;
        

elseif strcmp(HDR.TYPE,'EVENT') & any(lower(PERMISSION)=='w'),
	%%% Save event file in GDF-format
	HDR.TYPE = 'GDF';
	HDR.NS   = 0; 
	HDR.NRec = 0; 
	if any(isnan(HDR.T0))
		HDR.T0 = clock;
	end;
        HDR = sopen(HDR,'w');
	HDR = sclose(HDR);
	HDR.TYPE = 'EVENT';


elseif strcmp(HDR.TYPE,'BKR'),
        HDR = bkropen(HDR,PERMISSION);
        HDR.GDFTYP = repmat(3,1,HDR.NS);
        %%% Get trigger information from BKR data 

        
elseif strmatch(HDR.TYPE,{'CNT';'AVG';'EEG'})
        if any(PERMISSION=='r');
                [HDR,H1,h2] = cntopen(HDR,PERMISSION);
                if HDR.GDFTYP==3,       
                        % support of OVERFLOWDETECTION
                       HDR.THRESHOLD = repmat([-2^15,2^15-1],HDR.NS,1);
                end; 
                
        elseif any(PERMISSION=='w');
                % check header information
                if ~isfield(HDR,'NS'),
                        HDR.NS = 0;
                end;
                if ~isfinite(HDR.NS) | (HDR.NS<0)
                        fprintf(HDR.FILE.stderr,'Error SOPEN CNT-Write: HDR.NS not defined\n');
                        return;
                end;	
                if ~isfield(HDR,'SPR'),
                        HDR.SPR = 0;
                end;
                if ~isfinite(HDR.SPR)
                        HDR.SPR = 0;
                end;	
                type = 2;
                if strmatch(HDR.TYPE,'EEG'), type = 1;
                elseif strmatch(HDR.TYPE,'AVG'), type = 0;
                end;
                
                if ~isfield(HDR,'PID')
                        HDR.PID = char(repmat(32,1,20));
                elseif numel(HDR.PID)>20,
                        HDR.PID = HDR.PID(1:20);
                else 
                        HDR.PID = [HDR.PID(:)',repmat(32,1,20-length(HDR.PID(:)))];
                        %HDR.PID = [HDR.PID,repmat(32,1,20-length(HDR.PID))];
                end;
                
                if ~isfield(HDR,'Label')
                        HDR.Label = int2str((1:HDR.NS)');
                elseif iscell(HDR.Label),
                        HDR.Label = cat(1,HDR.Label);
                end;
                if size(HDR.Label,2)>10,
                        HDR.Label = HDR.Label(:,1:10);
                elseif size(HDR.Label,2)<10, 
                        HDR.Label = [HDR.Label,repmat(32,HDR.NS,10-size(HDR.Label,2))];
                end;
                
                if ~isfield(HDR,'Calib')
                        HDR.Cal = ones(HDR.NS,1);
                        e.sensitivity = ones(HDR.NS,1)*204.8;
                        HDR.Off = zeros(HDR.NS,1);
                else
                        HDR.Cal = diag(HDR.Calib(2:end,:));
                        e.sensitivity = ones(HDR.NS,1)*204.8;
                        HDR.Off = round(HDR.Calib(1,:)'./HDR.Cal);
                end;
                
                % open file 
                HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
                if HDR.FILE.FID < 0,
                        return;
                end;
                HDR.FILE.OPEN = 2; 
                if any([HDR.SPR] <= 0);
                        HDR.FILE.OPEN = 3; 
                end;
                
                % write fixed header
                fwrite(HDR.FILE.FID,'Version 3.0','char');
                fwrite(HDR.FILE.FID,zeros(2,1),'uint32');
                fwrite(HDR.FILE.FID,type,'uchar');
                fwrite(HDR.FILE.FID,HDR.PID,'uchar');
                
                fwrite(HDR.FILE.FID,repmat(0,1,900-ftell(HDR.FILE.FID)),'uchar')
                
                % write variable header
                for k = 1:HDR.NS,
                        count = fwrite(HDR.FILE.FID,HDR.Label(k,:),'uchar');
                        count = fwrite(HDR.FILE.FID,zeros(5,1),'uchar');
                        count = fwrite(HDR.FILE.FID, 0, 'ushort');
                        count = fwrite(HDR.FILE.FID,zeros(2,1),'uchar');
                        
                        count = fwrite(HDR.FILE.FID,zeros(7,1),'float');
                        count = fwrite(HDR.FILE.FID,HDR.Off(k),'short');
                        count = fwrite(HDR.FILE.FID,zeros(2,1),'uchar');
                        count = fwrite(HDR.FILE.FID,[zeros(2,1),e.sensitivity(k)],'float');
                        count = fwrite(HDR.FILE.FID,zeros(3,1),'char');
                        count = fwrite(HDR.FILE.FID,zeros(4,1),'uchar');
                        count = fwrite(HDR.FILE.FID,zeros(1,1),'char');
                        count = fwrite(HDR.FILE.FID,HDR.Cal(k),'short');
                end;	
                
                HDR.HeadLen = ftell(HDR.FILE.FID);
                if HDR.HeadLen ~= (900+75*HDR.NS),
                        fprintf(HDR.FILE.stderr,'Error SOPEN CNT-Write: Headersize does not fit\n');
                end;
        end;
        
        
elseif strcmp(HDR.TYPE,'FEF'),		% FEF/Vital format included
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,HDR.Endianity);
        status = fseek(HDR.FILE.FID,32,'bof'); 	% skip preamble
        
        if exist('fefopen','file') & ~status,
                HDR = fefopen(HDR);
        end;
        
        fprintf(HDR.FILE.stderr,'Warning SOPEN: Implementing Vital/FEF format not completed yet. Contact <a.schloegl@ieee.org> if you are interested in this feature.\n');
        HDR.FILE.FID = -1;
        return;        

        
elseif strcmp(HDR.TYPE,'SCP'),	%
        HDR = scpopen(HDR,PERMISSION,CHAN);
	if HDR.ERROR.status,
		fclose(HDR.FILE.FID);
		HDR.FILE.OPEN = 0; 
		return;
	end;	
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
	        
        
elseif strcmp(HDR.TYPE,'EBS'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-be');
        
        fprintf(HDR.FILE.stderr,'Warning SOPEN: Implementing EBS format not completed yet. Contact <a.schloegl@ieee.org> if you are interested in this feature.\n');
        
        %%%%% (1) Fixed Header (32 bytes) %%%%%
        HDR.VERSION = fread(HDR.FILE.FID,[1,8],'char');	%
        if strncmp(char(HDR.VERSION(1:3)),'EBS') 
                fprintf(HDR.FILE.stderr,'Error LOADEBS: %s not an EBS-File',HDR.FileName); 
                if any(HDR.VERSION(4:8)~=hex2dec(['94';'0a';'13';'1a';'0d'])'); 
                        fprintf(HDR.FILE.stderr,'Warning SOPEN EBS: %s may be corrupted',HDR.FileName); 
                end; 
        end;
        HDR.EncodingId = fread(HDR.FILE.FID,1,'int32');	%
        HDR.NS  = fread(HDR.FILE.FID,1,'uint32');	% Number of Channels
        HDR.SPR = fread(HDR.FILE.FID,2,'uint32');	% Number of Samples
        if HDR.SPR(1)==0,
                HDR.SPR=HDR.SPR(2);
        else 
                fprintf(HDR.FILE.stderr,'Error SOPEN: EBS-FILE %s too large',HDR.FileName); 
        end;
        LenData=fread(HDR.FILE.FID,1,'int64');	% Data Length
        
        %%%%% (2) LOAD Variable Header %%%%%
        tag=fread(HDR.FILE.FID,1,'int32');	% Tag field
        while (tag~=0),
                l  =fread(HDR.FILE.FID,1,'int32');	% length of value field
                val=char(fread(HDR.FILE.FID,4*l,'char')');	% depends on Tag field
                if     tag==hex2dec('00000002'),	%IGNORE
                elseif tag==hex2dec('00000004') HDR.PATIENT_NAME=val;
                elseif tag==hex2dec('00000006') HDR.PATIENT_ID=val;
                elseif tag==hex2dec('00000008') HDR.PATIENT_BIRTHDAY=val;
                elseif tag==hex2dec('0000000a') HDR.PATIENT_SEX=val;
                elseif tag==hex2dec('0000000c') HDR.SHORT_DESCRIPTION=val;
                elseif tag==hex2dec('0000000e') HDR.DESCRIPTION=val;
                elseif tag==hex2dec('00000010') HDR.SAMPLE_RATE=str2double(val);
                elseif tag==hex2dec('00000012') HDR.INSTITUTION=val;
                elseif tag==hex2dec('00000014') HDR.PROCESSING_HISTORY=val;
                elseif tag==hex2dec('00000016') HDR.LOCATION_DIAGRAM=val;
                        
                elseif tag==hex2dec('00000001') HDR.PREFERRED_INTEGER_RANGE=vec2matx(vec2matx(val,HDR.NS),2);
                elseif tag==hex2dec('00000003') HDR.PhysDim=val;
                elseif tag==hex2dec('00000005') HDR.CHANNEL_DESCRIPTION=val;
                elseif tag==hex2dec('00000007') HDR.CHANNEL_GROUPS=val;
                elseif tag==hex2dec('00000009') HDR.EVENTS=val;
                elseif tag==hex2dec('0000000b') HDR.RECORDING_TIME=val;
                elseif tag==hex2dec('0000000d') HDR.HDR.CHANNEL_LOCATIONS=val;
                elseif tag==hex2dec('0000000f') HDR.FILTERS=val;
                end;
                tag=fread(HDR.FILE.FID,1,'int32');	% Tag field
        end; 
        fclose(HDR.FILE.FID);
        
        
elseif strcmp(HDR.TYPE,'rhdE'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        
        %fseek(HDR.FILE.FID,4,'bof');		% skip 4 bytes ID
        %HDR.HeadLen = fread(HDR.FILE.FID,1,'int32');	% Length of Header ? 
        %HDR.H2 = fread(HDR.FILE.FID,5,'int32');	
        %HDR.NS = fread(HDR.FILE.FID,1,'int32');		% ? number of channels
        %HDR.H3 = fread(HDR.FILE.FID,5,'int32');	
        tmp = fread(HDR.FILE.FID,10,'int32');	
        HDR.HeadLen = tmp(2);		% Length of Header ? 
        HDR.H2 = tmp;
        HDR.NS = tmp(8);		% ? number of channels
        
        HDR.AS.endpos = HDR.FILE.size;
        HDR.NRec = (HDR.AS.endpos-HDR.HeadLen)/1024;
        
        fprintf(1,'Warning SOPEN HolterExcel2: is under construction.\n');
        
        if (nargout>1),	% just for testing
                H1 = fread(HDR.FILE.FID,[1,inf],'uchar')';
        end;
        fclose(HDR.FILE.FID);
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
        
        
elseif strcmp(HDR.TYPE,'alpha') & any(PERMISSION=='r'),
        HDR.FILE.FID = -1;      % make sure SLOAD does not call SREAD;

        % The header files are text files (not binary).
        try
                PERMISSION = 'rt';	% MatLAB default is binary, force Mode='rt';
                fid = fopen(fullfile(HDR.FILE.Path,'head'),PERMISSION);	
        catch
                PERMISSION = 'r';	% Octave 2.1.50 default is text, but does not support Mode='rt', 
                fid = fopen(fullfile(HDR.FILE.Path,'head'),PERMISSION);	
        end;
        
	cfiles = {'alpha.alp','eog','mkdef','r_info','rawhead','imp_res','sleep','../s_info'};
	%%,'marker','digin','montage','measure','cal_res'
	
	HDR.alpha = [];
	for k = 1:length(cfiles),
		[cf,tmp]=strtok(cfiles{k},'./');
    		fid = fopen(fullfile(HDR.FILE.Path,cfiles{k}),PERMISSION);
                if fid>0,
                        S = {};
                        state = 0; 
                        flag.rawhead = strcmp(cf,'rawhead');
                        flag.sleep = strcmp(cf,'sleep');
                        [s] = fgetl(fid);
                        while ischar(s), %~feof(fid),
                                [tag,s] = strtok(s,'= ');
                                s(find(s=='='))=' ';
                                [VAL,s1] = strtok(s,' ');
                                [val,status] = str2double(VAL);
                                if (state==0),
                                        if any(status),
                                                S=setfield(S,tag,VAL);
                                        else
                                                S=setfield(S,tag,val);
                                        end;
                                        if (flag.rawhead & strncmp(tag,'DispFlags',9))
                                                state = 1;
                                                k1 = 0; 
                                        elseif (flag.sleep & strncmp(tag,'SleepType',9))
                                                state = 3;
                                                k1 = 0; 
                                        end;
                                elseif (state==1)	% rawhead: channel info
                                        k1 = k1+1;
                                        HDR.Label{k1} = [tag,' ']; 
                                        [num,status,sa] = str2double(s);
                                        XY(k1,1:2) = num(4:5);
                                        CHANTYPE{k1}  = sa{3};
                                        HDR.alpha.chanidx(k1)   = num(2);
                                        if (k1==S.ChanCount);
                                                [tmp,HDR.alpha.chanorder]  = sort(HDR.alpha.chanidx);
                                                HDR.Label = HDR.Label(HDR.alpha.chanorder);
                                                XY = XY(HDR.alpha.chanorder,:);
                                                tmp = sum(XY.^2,2);
                                                HDR.ELEC.XYZ = [XY,sqrt(max(tmp)-tmp)];
                                                CHANTYPE  = CHANTYPE(HDR.alpha.chanorder);
                                                state = 2;
                                                k1 = 0; 
                                                HDR.alpha.chantyp.num = [];
                                        end;			
                                elseif (state==2)	% rawhead: info on channel type 
                                        k1 = k1+1; 
                                        [num,status,sa] = str2double(s,',');
                                        chantyp.s{k1}   = s;
                                        chantyp.tag{k1} = tag;
                                elseif (state==3)	% sleep: scoreing 
                                        k1 = k1+1; 
                                        scoring(k1) = val;
                                end
                                [s] = fgetl(fid);
                        end;	
                        fclose(fid);
                        HDR.alpha=setfield(HDR.alpha,cf,S);
                end;
	end;
        HDR.VERSION = HDR.alpha.rawhead.Version;
	if isfield(HDR.alpha,'rawhead')
		HDR.Bits = HDR.alpha.rawhead.BitsPerValue;
		HDR.NS   = HDR.alpha.rawhead.ChanCount;
		HDR.SampleRate = HDR.alpha.rawhead.SampleFreq;
		HDR.SPR  = HDR.alpha.rawhead.SampleCount;
		HDR.Filter.Notch = HDR.alpha.rawhead.NotchFreq;
                if     HDR.Bits == 12; HDR.GDFTYP = HDR.Bits+255;
                elseif HDR.Bits == 16; HDR.GDFTYP = 3;
                elseif HDR.Bits == 32; HDR.GDFTYP = 5;
                else   fprintf(HDR.FILE.stderr,'Error SOPEN(alpha): invalid header information.\n'); return;
                end;
                [datatyp, limits, datatypes] = gdfdatatype(HDR.GDFTYP);
                % THRESHOLD for Overflow detection
                HDR.THRESHOLD = repmat(limits, HDR.NS, 1);
		HDR.NRec = 1; 

		% channel-specific settings
		ix = zeros(1,HDR.NS);
		for k = 1:HDR.NS,
			ix(k) = strmatch(CHANTYPE{k},chantyp.tag,'exact');
		end;

		chantyp.s = chantyp.s(ix);
		for k = 1:HDR.NS,
			HDR.Filter.HighPass(k) = num(1);
			HDR.Filter.LowPass(k) = num(2);
			[num,status,sa] = str2double(chantyp.s{k},',');
			if strcmp(sa{5},'%%'); 
				sa{5}='%'; 
			end;
			HDR.PhysDim{k}=[deblank(sa{5}),' '];
		end;
		HDR.PhysDim = strvcat(HDR.PhysDim);
	else
                fprintf(HDR.FILE.stderr,'Error SOPEN (alpha): couldnot open RAWHEAD\n');
	end;
	if isfield(HDR.alpha,'sleep')
		HDR.alpha.sleep.scoring = scoring; 
	end;
	if isfield(HDR.alpha,'r_info')
                HDR.ID.Recording = HDR.alpha.r_info.RecId;
                HDR.ID.Lab = HDR.alpha.r_info.Laboratory;
		tmp = [HDR.alpha.r_info.RecDate,' ',HDR.alpha.r_info.RecTime];
		tmp(tmp=='.') = ' ';
		[tmp,status]=str2double(tmp);
		if ~any(status)
			HDR.T0 = tmp([3,2,1,4:6]);
		end;	
	end;
	if isfield(HDR.alpha,'s_info')
        %       HDR.Patient.Name = [HDR.alpha.s_info.LastName,', ',HDR.alpha.s_info.FirstName];
                HDR.Patient.Sex  = HDR.alpha.s_info.Gender;
                HDR.Patient.Sex  = HDR.alpha.s_info.Handedness;
                tmp = HDR.alpha.s_info.BirthDay;
                tmp(tmp=='.')=' ';
                t0 = str2double(tmp);
                age = [HDR.T0(1:3)-t0([3,2,1])]*[365.25;30;1]; % days 
                if (age<100)
                        HDR.Patient.Age = sprintf('%i day',age);
                elseif (age<1000)
                        HDR.Patient.Age = sprintf('%4.1f month',age/30);
                else
                        HDR.Patient.Age = sprintf('%4.1f year(s)',age/365.25);
                end;
	end;

        fid = fopen(fullfile(HDR.FILE.Path,'cal_res'),PERMISSION);
        if fid < 0,
                fprintf(HDR.FILE.stderr,'Warning SOPEN alpha-trace: could not open CAL_RES. Data is uncalibrated.\n');
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
                HDR.FLAG.UCAL = 1; 
        else
		k = 0; 
		while (k<2)		%% skip first 2 lines
	                [s] = fgetl(fid);   
			if ~strncmp(s,'#',1), %% comment lins do not count
				k=k+1; 
			end;
		end;	
		[s] = fread(fid,[1,inf],'uchar'); 
		fclose(fid);

		s(s=='=') = ',';
		[val,status,strarray]=str2double(s);
		
		HDR.Cal = val(HDR.alpha.chanorder,3);
		HDR.Off = val(HDR.alpha.chanorder,4);
                HDR.Label2 = strvcat(strarray(HDR.alpha.chanorder,1));
		OK = strmatch('no',strarray(HDR.alpha.chanorder,2));
		
                HDR.FLAG.UCAL = ~isempty(OK);
                if ~isempty(OK),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (alpha): calibration not valid for some channels\n');
                end;
                HDR.Cal(OK) = NaN;
                HDR.Calib = sparse([-HDR.Off';eye(HDR.NS*[1,1])])*sparse(1:HDR.NS,1:HDR.NS,HDR.Cal);
        end;        
        
        fid = fopen(fullfile(HDR.FILE.Path,'marker'),PERMISSION);
        if fid > 0,
		k = 0; 
		while (k<1)		%% skip first 2 lines
	                [s] = fgetl(fid);   
			if ~strncmp(s,'#',1), %% comment lins do not count
				k=k+1; 
			end;
		end;	
		[s] = fread(fid,[1,inf],'uchar'); 
		fclose(fid);

		s(s=='=') = ',';
		[val,status,strarray]=str2double(s,', ');
		HDR.EVENT.POS = val(:,3);
		[HDR.EVENT.CodeDesc,tmp,HDR.EVENT.TYP] = unique(strarray(:,1));
		ix = strmatch('off',strarray(:,2));
		HDR.EVENT.TYP(ix) = HDR.EVENT.TYP(ix)+hex2dec('8000'); 
        end;        
        
        fid = fopen(fullfile(HDR.FILE.Path,'montage'),PERMISSION);
        if fid > 0,
		K = 0; 
		while ~feof(fid),
	                s = fgetl(fid);   
			[tag,s]   = strtok(s,' =');
			[val1,r]  = strtok(s,' =,');
			if strncmp(tag,'Version',7),
			elseif strncmp(tag,'Montage',7),
				K = K+1;
				Montage{K,1} = s(4:end);
				k = 0;
			elseif strncmp(tag,'Trace',5),
				k = k+1; 
				trace{k,K} = s(4:end);
				[num,status,str] = str2double(s(4:end),[32,34,44]);
				if strcmpi(str{3},'xxx')
					Label{k,K} = str{2};
				else
					Label{k,K} = [str{2},'-',str{3}];
				end;	
			elseif strncmp(tag,'RefType',7),
			end;	
		end;	
		fclose(fid);
		HDR.alpha.montage.Montage = Montage;
		HDR.alpha.montage.Label   = Label;
		HDR.alpha.montage.trace   = trace;
        end;        
        
        fid = fopen(fullfile(HDR.FILE.Path,'digin'),PERMISSION);
        if 1,
	elseif fid < 0,
                fprintf(HDR.FILE.stderr,'Warning SOPEN alpha-trace: couldnot open DIGIN - no event information included\n');
        else
                [s] = fgetl(fid);       % read version
                
                k = 0; POS = []; DUR = []; TYP = []; IO = [];
                while ~feof(fid),
                        [s] = fgetl(fid);
                        if ~isnumeric(s),
                                [timestamp,s] = strtok(s,'='); 
                                [type,io] = strtok(s,'=,');
                                timestamp = str2double(timestamp);
                                if ~isnan(timestamp),
                                        k = k + 1;
                                        POS(k) = timestamp;     
                                        TYP(k) = hex2dec(type);   
                                        DUR(k) = 0;
                                        if (k>1) & (TYP(k)==0),
                                                DUR(k-1) = POS(k)-POS(k-1);
                                        end;
                                else
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN: alpha: invalid Event type\n');
                                end;	                        
                                if length(io)>1,
                                        IO(k) = io(2);
                                end;
                        end;
                end;
                fclose(fid);
                HDR.EVENT.N   = k;
                HDR.EVENT.POS = POS(:);
                HDR.EVENT.DUR = DUR(:);
                HDR.EVENT.TYP = TYP(:);
                HDR.EVENT.IO  = IO(:);
                HDR.EVENT.CHN = zeros(HDR.EVENT.N,1);
        end;
        if all(abs(HDR.alpha.rawhead.Version-[407.1,407.11,409.5]) > 1e-6);
                fprintf(HDR.FILE.stderr,'Warning SLOAD: Format ALPHA Version %6.2f not tested yet.\n',HDR.VERSION);
        end;
        
        HDR.FILE.FID = fopen(fullfile(HDR.FILE.Path,'rawdata'),'rb');
        if HDR.FILE.FID > 0,
                HDR.VERSION2  = fread(HDR.FILE.FID,1,'int16');
                HDR.NS   = fread(HDR.FILE.FID,1,'int16');
                HDR.Bits = fread(HDR.FILE.FID,1,'int16');
                HDR.AS.bpb = HDR.NS*HDR.Bits/8;
                HDR.SPR = 1; 
                if rem(HDR.AS.bpb,1),
                        HDR.AS.bpb = HDR.AS.bpb*2; %HDR.NS*HDR.Bits/8;
                        HDR.SPR = 2; 
                end;
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS  = 0;
                HDR.HeadLen = ftell(HDR.FILE.FID);
                fseek(HDR.FILE.FID,0,'eof');
                HDR.AS.endpos = (ftell(HDR.FILE.FID)-HDR.HeadLen)/HDR.AS.bpb;
                HDR.NRec = HDR.AS.endpos;
                fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
        end;
        
        
elseif strcmp(HDR.TYPE,'DEMG'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');     
        if ~isempty(findstr(PERMISSION,'r')),		%%%%% READ 
                % read header
                fseek(HDR.FILE.FID,4,'bof');    % skip first 4 bytes, should contain 'DEMG'
                HDR.VERSION = fread(HDR.FILE.FID,1,'uint16');	
                HDR.NS  = fread(HDR.FILE.FID,1,'uint16'); 
                HDR.SampleRate = fread(HDR.FILE.FID,1,'uint32');
                HDR.SPR = fread(HDR.FILE.FID,1,'uint32');
                HDR.NRec = 1; 
                
                HDR.Bits = fread(HDR.FILE.FID,1,'uint8');
                HDR.PhysMin = fread(HDR.FILE.FID,1,'int8');
                HDR.PhysMax = fread(HDR.FILE.FID,1,'int8');
                if HDR.VERSION==1,
                        HDR.GDFTYP = 'float32';
                        HDR.Cal = 1; 
                        HDR.Off = 0; 
                        HDR.AS.bpb = 4*HDR.NS;
                elseif HDR.VERSION==2,
                        HDR.GDFTYP = 'uint16';
                        HDR.Cal = (HDR.PhysMax-HDR.PhysMin)/(2^HDR.Bits-1);
                        HDR.Off = HDR.PhysMin;
                        HDR.AS.bpb = 2*HDR.NS;
                else    
                        fprintf(HDR.FILE.stderr,'Error SOPEN DEMG: invalid version number.\n');
                        fclose(HDR.FILE.FID);
                        HDR.FILE.FID=-1;
                        return;
                end;
                HDR.Calib = sparse([ones(1,HDR.NS),2:HDR.NS+1],[1:HDR.NS,1:HDR.NS],ones(HDR.NS,1)*[HDR.Off,HDR.Cal],HDR.NS+1,HDR.NS);
                HDR.HeadLen = ftell(HDR.FILE.FID);
                HDR.FILE.POS = 0;
                HDR.FILE.OPEN = 1; 
                HDR.AS.endpos = HDR.SPR;
                %HDR.Filter.LowPass = 450;       % default values
                %HDR.Filter.HighPass = 20;       % default values
                
        else
                fprintf(HDR.FILE.stderr,'Warning SOPEN DEMG: writing not implemented, yet.\n');
        end;
        
        
elseif strcmp(HDR.TYPE,'ACQ'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        
        %--------    Fixed Header        
        ItemHeaderLen = fread(HDR.FILE.FID,1,'uint16');
        HDR.VERSION = fread(HDR.FILE.FID,1,'uint32');
        HDR.ACQ.ExtItemHeaderLen = fread(HDR.FILE.FID,1,'uint32');

        HDR.NS = fread(HDR.FILE.FID,1,'int16');
        HDR.ACQ.HorizAxisType = fread(HDR.FILE.FID,1,'int16');
        HDR.ACQ.CurChannel = fread(HDR.FILE.FID,1,'int16');
        HDR.ACQ.SampleTime = fread(HDR.FILE.FID,1,'float64')/1000;
        HDR.SampleRate = 1/HDR.ACQ.SampleTime;
        HDR.TimeOffset = fread(HDR.FILE.FID,1,'float64')/1000;
        HDR.TimeScale  = fread(HDR.FILE.FID,1,'float64');
        HDR.ACQ.TimeCursor1  = fread(HDR.FILE.FID,1,'float64');
        HDR.ACQ.TimeCursor2  = fread(HDR.FILE.FID,1,'float64');
        HDR.ACQ.rcWindow  = fread(HDR.FILE.FID,1,'float64');
        HDR.ACQ.MeasurementType = fread(HDR.FILE.FID,6,'int16');
        HDR.ACQ.HiLite    = fread(HDR.FILE.FID,2,'uint8');
        HDR.FirstTimeOffset = fread(HDR.FILE.FID,1,'float64');
        
        fseek(HDR.FILE.FID,HDR.ACQ.ExtItemHeaderLen,'bof');

        % --------   Variable Header        
        
        % --------   Per Channel data section 
        HDR.Labels = char(zeros(HDR.NS,40));
        HDR.Off = zeros(HDR.NS,1);
        HDR.Cal = ones(HDR.NS,1);
        HDR.ChanHeaderLen = zeros(HDR.NS,1);
        HDR.PhysDim = char(zeros(HDR.NS,20));
        offset = ftell(HDR.FILE.FID); 
        for k = 1:HDR.NS;
                fseek(HDR.FILE.FID,offset+sum(HDR.ChanHeaderLen),'bof');
                HDR.ChanHeaderLen(k) = fread(HDR.FILE.FID,1,'uint32');
                HDR.ChanSel(k) = fread(HDR.FILE.FID,1,'int16');
                HDR.Label(k,1:40) = fread(HDR.FILE.FID,[1,40],'char');
                rgbColor = fread(HDR.FILE.FID,4,'int8');
                DispChan = fread(HDR.FILE.FID,2,'int8');
                HDR.Off(k) = fread(HDR.FILE.FID,1,'float64');
                HDR.Cal(k) = fread(HDR.FILE.FID,1,'float64');
                HDR.PhysDim(k,1:20) = fread(HDR.FILE.FID,[1,20],'char');
                HDR.ACQ.BufLength(k) = fread(HDR.FILE.FID,1,'int32');
                HDR.AmpGain(k) = fread(HDR.FILE.FID,1,'float64');
                HDR.AmpOff(k) = fread(HDR.FILE.FID,1,'float64');
                HDR.ACQ.ChanOrder = fread(HDR.FILE.FID,1,'int16');
                HDR.ACQ.DispSize = fread(HDR.FILE.FID,1,'int16');
                
                if HDR.VERSION >= 34,
                        fseek(HDR.FILE.FID,10,'cof');
                end;
                if HDR.VERSION >= 38,   % version of Acq 3.7.0-3.7.2 (Win 98, 98SE, NT, Me, 2000) and above
                        HDR.Description(k,1:128) = fread(HDR.FILE.FID,[1,128],'char');
                        HDR.ACQ.VarSampleDiv(k) = fread(HDR.FILE.FID,1,'uint16');
                else
                        HDR.ACQ.VarSampleDiv(k) = 1;
                end;
                if HDR.VERSION >= 39,  % version of Acq 3.7.3 or above (Win 98, 98SE, 2000, Me, XP)
                        HDR.ACQ.VertPrecision(k) = fread(HDR.FILE.FID,1,'uint16');
                end;
        end;
        HDR.Label = char(HDR.Label);
        HDR.PhysDim = char(HDR.PhysDim);
        HDR.Calib = [HDR.Off(:).';diag(HDR.Cal)];
        HDR.SPR = HDR.ACQ.VarSampleDiv(1);
        for k = 2:length(HDR.ACQ.VarSampleDiv);
                HDR.SPR = lcm(HDR.SPR,HDR.ACQ.VarSampleDiv(k));
        end;
        HDR.NRec =  floor(min(HDR.ACQ.BufLength.*HDR.ACQ.VarSampleDiv/HDR.SPR)); 
        HDR.AS.SPR = HDR.SPR./HDR.ACQ.VarSampleDiv'; 
        HDR.AS.spb = sum(HDR.AS.SPR);	% Samples per Block
        HDR.AS.bi = [0;cumsum(HDR.AS.SPR(:))]; 
        HDR.ACQ.SampleRate = 1./(HDR.AS.SPR*HDR.ACQ.SampleTime);
        HDR.SampleRate = 1/HDR.ACQ.SampleTime;
        HDR.Dur = HDR.SPR*HDR.ACQ.SampleTime;
        
        %--------   foreign data section
        ForeignDataLength = fread(HDR.FILE.FID,1,'int16');
        HDR.ACQ.ForeignDataID = fread(HDR.FILE.FID,1,'uint16');
        HDR.ACQ.ForeignData = fread(HDR.FILE.FID,[1,ForeignDataLength-4],'char');
        %fseek(HDR.FILE.FID,ForeignDataLength-2,'cof');
        
        %--------   per channel data type section
        offset3 = 0;
        HDR.AS.bpb = 0;	
        for k = 1:HDR.NS,
                sz = fread(HDR.FILE.FID,1,'uint16');
                HDR.AS.bpb = HDR.AS.bpb + HDR.AS.SPR(k)*sz; 
                offset3 = offset3 + HDR.ACQ.BufLength(k) * sz;
                
                typ = fread(HDR.FILE.FID,1,'uint16');
                if ~any(typ==[1,2])
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (ACQ): invalid or unknonw data type in file %s.\n',HDR.FileName);
                end;
                HDR.GDFTYP(k) = 31-typ*14;   % 1 = int16; 2 = double
        end;
        HDR.AS.BPR  = ceil(HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)'); 
        while any(HDR.AS.BPR  ~= HDR.AS.SPR.*GDFTYP_BYTE(HDR.GDFTYP+1)');
                fprintf(2,'\nError SOPEN (ACQ): block configuration in file %s not supported.\n',HDR.FileName);
        end;
        
        % prepare SREAD for different data types 
        n = 0; 
        typ = [-1;HDR.GDFTYP(:)];
        for k = 1:HDR.NS; 
                if (typ(k) == typ(k+1)),
                        HDR.AS.c(n)   = HDR.AS.c(n)  + HDR.AS.SPR(k);
                        HDR.AS.c2(n)  = HDR.AS.c2(n) + HDR.AS.BPR(k);
                else
                        n = n + 1; 
                        HDR.AS.c(n)   = HDR.AS.SPR(k);
                        HDR.AS.c2(n)  = HDR.AS.BPR(k);
                        HDR.AS.TYP(n) = HDR.GDFTYP(k);
                end;
        end;
        
        HDR.HeadLen = HDR.ACQ.ExtItemHeaderLen + sum(HDR.ChanHeaderLen) + ForeignDataLength + 4*HDR.NS; 
        HDR.FILE.POS = 0; 
        HDR.FILE.OPEN = 1; 
        HDR.AS.endpos = HDR.HeadLen + offset3; 
        fseek(HDR.FILE.FID,HDR.AS.endpos,'bof');	

        %--------  Markers Header section
        len = fread(HDR.FILE.FID,1,'uint32');
        EVENT.N = fread(HDR.FILE.FID,1,'uint32');
        HDR.EVENT.POS = repmat(nan, EVENT.N ,1);
        HDR.EVENT.TYP = repmat(nan, EVENT.N ,1);

        for k = 1:EVENT.N, 
                %HDR.Event(k).Sample = fread(HDR.FILE.FID,1,'int32');
                HDR.EVENT.POS(k) = fread(HDR.FILE.FID,1,'int32');
                tmp = fread(HDR.FILE.FID,4,'uint16');
                HDR.Event(k).selected = tmp(1); 
                HDR.Event(k).TextLocked = tmp(2); 
                HDR.Event(k).PositionLocked = tmp(3); 
                textlen = tmp(4);
                HDR.Event(k).Text = fread(HDR.FILE.FID,textlen,'char');
        end;
        fseek(HDR.FILE.FID,HDR.HeadLen,'bof');	
        
        
elseif strncmp(HDR.TYPE,'AKO',3),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        HDR.Header = fread(HDR.FILE.FID,[1,46],'char');
        warning('support of AKO format not completed');
        HDR.Patient.ID = char(HDR.Header(17:24));
        HDR.SampleRate = 128; % ???
        HDR.NS = 1;
        HDR.NRec = 1; 
        HDR.Calib = [-127;1];
        [HDR.data,HDR.SPR] = fread(HDR.FILE.FID,inf,'uint8');
        fclose(HDR.FILE.FID);
        HDR.FILE.POS = 0;
        HDR.TYPE = 'native';
        
        
elseif strcmp(HDR.TYPE,'ALICE4'),
        fprintf(HDR.FILE.stderr,'Warning SOPEN: Support of ALICE4 format not completeted. \n\tCalibration, filter setttings and SamplingRate are missing\n');
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        [s,c]  = fread(HDR.FILE.FID,[1,408],'char');
        HDR.NS = s(55:56)*[1;256];
        HDR.SampleRate   = 100; 
        HDR.Patient.ID   = char(s(143:184));
        HDR.Patient.Sex  = char(s(185));
        HDR.Patient.Date = char(s(187:194));
        [H2,c] = fread(HDR.FILE.FID,[118,HDR.NS],'char');
        HDR.Label = char(H2(1:12,:)');
        HDR.HeadLen = ftell(HDR.FILE.FID);
        HDR.AS.bpb = HDR.NS*HDR.SampleRate + 5; 
        [a,count] = fread(HDR.FILE.FID,[HDR.AS.bpb,floor((HDR.FILE.size-HDR.HeadLen)/HDR.AS.bpb)],'uint8');
        fclose(HDR.FILE.FID);
        count = ceil(count/HDR.AS.bpb);
        HDR.data = repmat(NaN,100*count,HDR.NS);
        for k = 1:HDR.NS,
                HDR.data(:,k)=reshape(a(k*HDR.SampleRate+[1-HDR.SampleRate:0],:),HDR.SampleRate*count,1);
        end
        HDR.SPR = size(HDR.data,1);

        HDR.NRec = 1; 
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
        HDR.FLAG.UCAL = 1; 
        HDR.TYPE  = 'native';
        HDR.FILE.POS = 0;
        
        
elseif strcmp(HDR.TYPE,'ATES'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        HDR.Header = fread(HDR.FILE.FID,[1,128],'char');
        tmp = fread(HDR.FILE.FID,1,'int16');
        HDR.FLAG.Monopolar = logical(tmp);
        HDR.SampleRate = fread(HDR.FILE.FID,1,'int16');
        HDR.Cal = fread(HDR.FILE.FID,1,'float32');
        type = fread(HDR.FILE.FID,1,'float32');
        if type==2,
                HDR.GDFTYP = 'int16';
        else
                error('ATES: unknown type');
        end;
        HDR.ATES.Mask = fread(HDR.FILE.FID,2,'uint32');
        HDR.DigMax = fread(HDR.FILE.FID,1,'uint16');
        HDR.Filter.Notch = fread(HDR.FILE.FID,1,'uint16');
        HDR.SPR = fread(HDR.FILE.FID,1,'uint32');
        HDR.ATES.MontageName = fread(HDR.FILE.FID,8,'uchar');
        HDR.ATES.MontageComment = fread(HDR.FILE.FID,31,'uchar');
        HDR.NS = fread(HDR.FILE.FID,1,'int16');
        fclose(HDR.FILE.FID);

        
elseif strcmp(HDR.TYPE,'BLSC'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        HDR.Header = fread(HDR.FILE.FID,[1,3720],'char');       % ???
        HDR.data   = fread(HDR.FILE.FID,[32,inf],'ubit8');      % ???
        %HDR.NS = 32; 
        %HDR.SPR = 24063;
        fclose(HDR.FILE.FID);
        fprintf(2,'Error SOPEN: Format BLSC not supported (yet).\n'); 
        return; 

        
elseif strcmp(HDR.TYPE,'RigSys'),       % thanks to  J. Chen
        HDR.FILE.FID = fopen(HDR.FileName,'r','ieee-le');
        [thdr,count] = fread(HDR.FILE.FID,[1,1024],'char');
        thdr = char(thdr);
        HDR.RigSys.H1 = thdr;        
        empty_char = NaN; 
        STOPFLAG = 1; 
        while (STOPFLAG & ~isempty(thdr));
                [tline, thdr] = strtok(thdr,[13,10,0]);
                [tag, value]  = strtok(tline,'=');
                value = strtok(value,'=');
                if strcmp(tag,'FORMAT ISSUE'), 
                        HDR.VERSION = value; 
                elseif strcmp(tag,'EMPTY HEADER CHARACTER'), 
                        [t,v]=strtok(value);
                        if strcmp(t,'ASCII')
                                empty_char = str2double(v);
                                STOPFLAG = 0; 
                        else
                                fprintf(HDR.FILE.stderr,'Warning SOPEN (RigSys): Couldnot identify empty character');;
                        end;
                end;
        end                        
        if ~isfield(HDR,'VERSION')
                fprintf(HDR.FILE.stderr,'Error SOPEN (RigSys): could not open file %s. Specification not known.\n',HDR.FileName);
                HDR.TYPE = 'unknown';
                fclose(HDR.FILE.FID);
                return;
        end;
        [thdr,H1] = strtok(thdr,empty_char);
        while ~isempty(thdr);
                [tline, thdr] = strtok(thdr,[13,10,0]);
                [tag, value]  = strtok(tline,'=');
                value = strtok(value,'=');
                if 0, 
                elseif strcmp(tag,'HEADER SIZE'), 
                        HDR.RigSys.H1size = str2double(value);
                        if    count == HDR.RigSys.H1size,
                        elseif count < HDR.RigSys.H1size,
                                tmp = fread(HDR.FILE.FID,[1,HDR.RigSys.H1size-count],'char');
                                thdr = [thdr,char(tmp)];
                        elseif count > HDR.RigSys.H1size,
                                status = fseek(HDR.FILE.FID,HDR.RigSys.H1size);
                        end;        
                elseif strcmp(tag,'CHANNEL HEADER SIZE'), 
                        HDR.RigSys.H2size = str2double(value);
                elseif strcmp(tag,'FRAME HEADER SIZE'), 
                        HDR.RigSys.H3size = str2double(value);
                elseif strcmp(tag,'NCHANNELS'), 
                        HDR.NS = str2double(value);
                elseif strcmp(tag,'SAMPLE INTERVAL'), 
                        HDR.SampleRate = 1/str2double(value);
                elseif strcmp(tag,'HISTORY LENGTH'), 
                        HDR.AS.endpos = str2double(value);
                elseif strcmp(tag,'REFERENCE TIME'), 
                        HDR.RigSys.TO=value;
                        HDR.T0(1:6) = round(datevec(value)*1e4)*1e-4;
                end
        end;                                
        HDR.HeadLen = HDR.RigSys.H1size+HDR.RigSys.H1size*HDR.NS;
        
        [H1,c] = fread(HDR.FILE.FID,[HDR.RigSys.H2size,HDR.NS],'char');
        for chan=1:HDR.NS,
                [thdr] = strtok(char(H1(:,chan)'),empty_char);
                while ~isempty(thdr);
                        [tline, thdr] = strtok(thdr,[13,10,0]);
                        [tag, value]  = strtok(tline,'=');
                        value = strtok(value,'=');
                        if strcmp(tag,'FULL SCALE'), 
                                HDR.Gain(chan,1) = str2double(value); 
                        elseif strcmp(tag,'UNITS'), 
                                HDR.PhysDim{chan} = [value,' ']; 
                        elseif strcmp(tag,'OFFSET'), 
                                HDR.Off(chan) = str2double(value); 
                        elseif 0, strcmp(tag,'CHANNEL DESCRIPTION'), 
                                HDR.Label{chan} = [value,' ']; 
                        elseif strcmp(tag,'CHANNEL NAME'), 
                                HDR.Label{chan} = [value,' ']; 
                        elseif strcmp(tag,'SAMPLES PER BLOCK'), 
                                HDR.AS.SPR(chan) = str2double(value); 
                        elseif strcmp(tag,'BYTES PER SAMPLE'), 
                                HDR.Bits(chan) = str2double(value)*8; 
                        end;          
                end;
        end;
        fhsz = HDR.RigSys.H3size*8/HDR.Bits(1);
        s = fread(HDR.FILE.FID,[1024*HDR.NS+fhsz,inf],'int16');
        fclose(HDR.FILE.FID);
        HDR.RigSys.FrameHeaders=s(1:12,:);

        for k=1:HDR.NS,
                if k==1, HDR.SPR = HDR.AS.SPR(1);
                else HDR.SPR = lcm(HDR.SPR, HDR.AS.SPR(1));
                end;
        end;
        HDR.AS.bi = [0;cumsum(HDR.AS.SPR(:))];
        HDR.NRec = size(s,2); 
        HDR.FLAG.TRIGGERED = 0; 
        HDR.data = zeros(HDR.MAXSPR*HDR.NRec,HDR.NS);
        for k = 1:HDR.NS,
                tmp = s(fhsz+[HDR.AS.bi(k)+1:HDR.AS.bi(k+1)],:);
                HDR.data(:,k) = rs(tmp(:),1,HDR.MAXSPR/HDR.AS.SPR(k));
        end;
        HDR.data  = HDR.data(1:HDR.AS.endpos,:);
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Gain(:)./16384);

        HDR.Label    = strvcat(HDR.Label);
        HDR.PhysDim  = strvcat(HDR.PhysDim);
        HDR.FILE.POS = 0; 
        HDR.TYPE     = 'native';

        
elseif strcmp(HDR.TYPE,'SND'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,HDR.Endianity);
        if HDR.FILE.FID < 0,
                return;
        end;
        
        if ~isempty(findstr(PERMISSION,'r')),	%%%%% READ 
                HDR.FILE.OPEN = 1; 
                fseek(HDR.FILE.FID,4,'bof');
                HDR.HeadLen = fread(HDR.FILE.FID,1,'uint32');
                datlen = fread(HDR.FILE.FID,1,'uint32');
                HDR.FILE.TYPE = fread(HDR.FILE.FID,1,'uint32');
                HDR.SampleRate = fread(HDR.FILE.FID,1,'uint32');
                HDR.NS = fread(HDR.FILE.FID,1,'uint32');
		HDR.Label = repmat(' ',HDR.NS,1);
                [tmp,count] = fread(HDR.FILE.FID, [1,HDR.HeadLen-24],'char');
                HDR.INFO = setstr(tmp);
                
        elseif ~isempty(findstr(PERMISSION,'w')),	%%%%% WRITE 
                HDR.FILE.OPEN = 2; 
                if ~isfield(HDR,'NS'),
                        HDR.NS = 0;
                end;
                if ~isfield(HDR,'SPR'),
                        HDR.SPR = 0;
                end;
                if ~isfinite(HDR.NS)
                        HDR.NS = 0;
                end;	
                if ~isfinite(HDR.SPR)
                        HDR.SPR = 0;
                end;	
                if any([HDR.SPR,HDR.NS] <= 0);
                        HDR.FILE.OPEN = 3; 
                end;
                if ~isfield(HDR,'INFO')
                        HDR.INFO = HDR.FileName;
                end;
                len = length(HDR.INFO);
                if len == 0; 
                        HDR.INFO = 'INFO';
                else
                        HDR.INFO = [HDR.INFO,repmat(' ',1,mod(4-len,4))];	
                end;
                HDR.HeadLen = 24+length(HDR.INFO); 
                if ~isfield(HDR.FILE,'TYPE')
                        HDR.FILE.TYPE = 6; % default float32
                end;		
        end;
        
        if HDR.FILE.TYPE==1, 
                HDR.GDFTYP =  'uchar';
                HDR.Bits   =  8;
        elseif HDR.FILE.TYPE==2, 
                HDR.GDFTYP =  'int8';
                HDR.Bits   =  8;
        elseif HDR.FILE.TYPE==3, 
                HDR.GDFTYP =  'int16';
                HDR.Bits   = 16;
        elseif HDR.FILE.TYPE==4, 
                HDR.GDFTYP = 'bit24';
                HDR.Bits   = 24;
        elseif HDR.FILE.TYPE==5, 
                HDR.GDFTYP = 'int32';
                HDR.Bits   = 32;
        elseif HDR.FILE.TYPE==6, 
                HDR.GDFTYP = 'float32';
                HDR.Bits   = 32;
        elseif HDR.FILE.TYPE==7, 
                HDR.GDFTYP = 'float64';
                HDR.Bits   = 64;
                
        elseif HDR.FILE.TYPE==11, 
                HDR.GDFTYP = 'uint8';
                HDR.Bits   =  8;
        elseif HDR.FILE.TYPE==12, 
                HDR.GDFTYP = 'uint16';
                HDR.Bits   = 16;
        elseif HDR.FILE.TYPE==13, 
                HDR.GDFTYP = 'ubit24';
                HDR.Bits   = 24;
        elseif HDR.FILE.TYPE==14, 
                HDR.GDFTYP = 'uint32';
                HDR.Bits   = 32;
                
        else
                fprintf(HDR.FILE.stderr,'Error SOPEN SND-format: datatype %i not supported\n',HDR.FILE.TYPE);
                return;
        end;
	[d,l,d1,b,HDR.GDFTYP] = gdfdatatype(HDR.GDFTYP);
        HDR.AS.bpb = HDR.NS*HDR.Bits/8;
        
        % Calibration 
        if any(HDR.FILE.TYPE==[2:5]), 
                HDR.Cal = 2^(1-HDR.Bits); 
        else
                HDR.Cal = 1; 	
        end;
        HDR.Off = 0;
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal);
        
        %%%%% READ 
        if HDR.FILE.OPEN == 1; 
                % check file length
                fseek(HDR.FILE.FID,0,1);
                len = ftell(HDR.FILE.FID); 
                if len ~= (datlen+HDR.HeadLen),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN SND-format: header information does not fit file length \n');
                        datlen = len - HDR.HeadLen; 
                end;	
                fseek(HDR.FILE.FID,HDR.HeadLen,-1);
                HDR.SPR  = datlen/HDR.AS.bpb;
                HDR.AS.endpos = datlen/HDR.AS.bpb;
                HDR.Dur  = HDR.SPR/HDR.SampleRate;
                
                
                %%%%% WRITE 
        elseif HDR.FILE.OPEN > 1; 
                datlen = HDR.SPR * HDR.AS.bpb;
                fwrite(HDR.FILE.FID,[hex2dec('2e736e64'),HDR.HeadLen,datlen,HDR.FILE.TYPE,HDR.SampleRate,HDR.NS],'uint32');
                fwrite(HDR.FILE.FID,HDR.INFO,'char');
                
        end;
        HDR.FILE.POS = 0;
        HDR.NRec = 1;
        
        
elseif strncmp(HDR.TYPE,'EEG-1100',8),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        if any(PERMISSION=='r'),		%%%%% READ 
                [H1,count] = fread(HDR.FILE.FID,[1,6160],'char');
                %HDR.Patient.Name = char(H1(79+(1:32)));
                if count<6160, 
                        fclose(HDR.FILE.FID);
                        return;
                end;
                HDR.T0(1:6) = str2double({H1(65:68),H1(69:70),H1(71:72),H1(6148:6149),H1(6150:6151),H1(6152:6153)});
                if strcmp(HDR.FILE.Ext,'LOG')
                        [s,c] = fread(HDR.FILE.FID,[1,inf],'char');
                        s = char([H1(1025:end),s]);
                        K = 0; 
                        [t1,s] = strtok(s,0);
                        while ~isempty(s),
                                K = K + 1; 
                                [HDR.EVENT.x{K},s] = strtok(s,0);
                        end	
                end;
                fclose(HDR.FILE.FID);
        end;
        
        
elseif strcmp(HDR.TYPE,'GTF'),          % Galileo EBNeuro EEG Trace File
%        HDR = gtfopen(HDR,PERMISSION);

        HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
        
        % read 3 header blocks 
        HDR.GTF.H1 = fread(HDR.FILE.FID,[1,512],'uint8');
        HDR.GTF.H2 = fread(HDR.FILE.FID,[1,15306],'int8');
        HDR.GTF.H3 = fread(HDR.FILE.FID,[1,8146],'uint8');           
        HDR.GTF.messages = fread(HDR.FILE.FID,[3600,1],'int8');
        HDR.GTF.states   = fread(HDR.FILE.FID,[3600,1],'int8');
        
        HDR.GTF.L1 = char(reshape(HDR.GTF.H3(1:650),65,10)');                   
        HDR.GTF.L2 = char(reshape(HDR.GTF.H3(650+(1:20*16)),16,20)');
        HDR.GTF.L3 = reshape(HDR.GTF.H3(1070+32*3+(1:232*20)),232,20)';
        
        HDR.Label = char(reshape(HDR.GTF.H3(1071:1070+32*3),3,32)');        % channel labels

        [H.i8, count]    = fread(HDR.FILE.FID,inf,'int8');
        fclose(HDR.FILE.FID);
        
        [t,status] = str2double(char([HDR.GTF.H1(35:36),32,HDR.GTF.H1(37:39)]));	
        if ~any(status) & all(t>0)
                HDR.NS = t(1); 
                HDR.SampleRate = t(2); 
        else
                fprintf(2,'ERROR SOPEN (%s): Invalid GTF header.\n',HDR.FileName);
					 HDR.TYPE = 'unknown';
                return; 
        end

        % convert messages, states and annotations into EVENT's
        ix = find(HDR.GTF.messages<-1);
        ann.POS  = ix*HDR.SampleRate;
        ann.TYP  = -HDR.GTF.messages(ix)-1;
        ann.Desc = [repmat('A: ',length(ix),1),HDR.GTF.L1(ann.TYP,:)];
        ix = find(HDR.GTF.messages>-1);
        msg.POS  = ix*HDR.SampleRate;
        msg.TYP  = 1+HDR.GTF.messages(ix);
        msg.Desc = [repmat('M: ',length(ix),1),HDR.GTF.L2(msg.TYP,:)];
        ix       = find((HDR.GTF.states>9) & (HDR.GTF.states<20));
        sts.POS  = ix*HDR.SampleRate;
        sts.TYP  = HDR.GTF.states(ix)+1;
        % ix       = find((HDR.GTF.states==20));  % Calibration ??? 
        
        sts.Desc = [repmat('S: ',length(ix),1),HDR.GTF.L2(sts.TYP,:)];
        HDR.EVENT.POS  = [ann.POS(:); msg.POS(:); sts.POS(:)];
        HDR.EVENT.TYP  = [ann.TYP(:); msg.TYP(:)+10; sts.TYP(:)+10];
        HDR.EVENT.Desc = cellstr(strvcat(ann.Desc,msg.Desc,sts.Desc));
        
        HDR.GTF.ann = ann; 
        HDR.GTF.msg = msg; 
        HDR.GTF.sts = sts; 
        
        HDR.Dur  = 10; 
        HDR.SPR  = HDR.Dur*HDR.SampleRate; 
        HDR.Bits = 8; 
        HDR.GDFTYP = repmat(1,HDR.NS,1);
        HDR.TYPE = 'native'; 
        HDR.THRESHOLD = repmat([-127,127],HDR.NS,1);    % support of overflow detection
        HDR.FILE.POS = 0; 
        HDR.Label = HDR.Label(1:HDR.NS,:);
        
        HDR.AS.bpb = (HDR.SampleRate*240+2048);
        HDR.GTF.Preset = HDR.GTF.H3(8134)+1;	% Preset

        t2 = (0:floor(count/HDR.AS.bpb)-1)*HDR.AS.bpb;
        HDR.NRec = length(t2);
        [s2,sz]  = trigg(H.i8,t2+2048,1,HDR.SampleRate*240);
        HDR.data = reshape(s2,[HDR.NS,sz(2)/HDR.NS*HDR.NRec])';
        
        [s4,sz]  = trigg(H.i8,t2+1963,0,1);
        sz(sz==1)= []; 
        x  = reshape(s4,sz)';   
        HDR.GTF.timestamp = (x+(x<0)*256)*[1;256];      % convert from 2*int8 in 1*uint16
        
	[s4,sz] = trigg(H.i8,t2,1,2048);
	sz(sz==1)= []; if length(sz)<2,sz = [sz,1]; end;
	s4 = reshape(s4,sz);

	tau  = [0.01, 0.03, 0.1, 0.3, 1];
	LowPass = [30, 70];

	%% Scaling 
	Sens = [.5, .7, 1, 1.4, 2, 5, 7, 10, 14, 20, 50, 70, 100, 140, 200]; 
	x    = reshape(s4(13:6:1932,:),32,HDR.NRec*HDR.Dur);
	Cal  = Sens(x(1:HDR.NS,:)+1)'/4;
	HDR.data  = HDR.data.*Cal(ceil((1:HDR.SampleRate*HDR.NRec*HDR.Dur)/HDR.SampleRate),:);
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
	HDR.PhysDim = 'uV';

                
elseif strcmp(HDR.TYPE,'MFER'),
        HDR = mwfopen(HDR,PERMISSION);
        if (HDR.FRAME.N ~= 1),
                fprintf(HDR.FILE.stderr,'Error SOPEN (MFER): files with more than one frame not implemented, yet.\n');
                fclose(HDR.FILE.FID);
                HDR.FILE.FID  =-1;
                HDR.FILE.OPEN = 0;
        end
        
        
elseif strcmp(HDR.TYPE,'MPEG'),
        % http://www.dv.co.yu/mpgscript/mpeghdr.htm
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        if ~isempty(findstr(PERMISSION,'r')),		%%%%% READ 
                % read header
                try,
                        tmp = fread(HDR.FILE.FID,1,'ubit11');
                catch
                        fprintf(HDR.FILE.stderr,'Error 1003 SOPEN: datatype UBIT11 not implented. Header cannot be read.\n');
                        return;
                end;
                HDR.MPEG.syncword = tmp;
                HDR.MPEG.ID = fread(HDR.FILE.FID,1,'ubit2');
                HDR.MPEG.layer = fread(HDR.FILE.FID,1,'ubit2');
                HDR.MPEG.protection_bit = fread(HDR.FILE.FID,1,'ubit1');
                HDR.MPEG.bitrate_index = fread(HDR.FILE.FID,1,'ubit4');
                HDR.MPEG.sampling_frequency_index = fread(HDR.FILE.FID,1,'ubit2');
                HDR.MPEG.padding_bit = fread(HDR.FILE.FID,1,'ubit1');
                HDR.MPEG.privat_bit = fread(HDR.FILE.FID,1,'ubit1');
                HDR.MPEG.mode = fread(HDR.FILE.FID,1,'ubit2');
                HDR.MPEG.mode_extension = fread(HDR.FILE.FID,1,'ubit2');
                HDR.MPEG.copyright = fread(HDR.FILE.FID,1,'ubit1');
                HDR.MPEG.original_home = fread(HDR.FILE.FID,1,'ubit1');
                HDR.MPEG.emphasis = fread(HDR.FILE.FID,1,'ubit2');
                
                switch HDR.MPEG.ID,	%Layer 
                        case 0,
                                HDR.VERSION = 2.5;
                        case 1,
                                HDR.VERSION = -1;% reserved
                        case 2,
                                HDR.VERSION = 2;
                        case 3,
                                HDR.VERSION = 1;
                end;
                
                tmp = [32,32,32,32,8; 64,48,40,48,16; 96,56,48,56,24; 128,64,56,64,32; 160,80,64,80,40; 192,96,80,96,48; 224,112,96,112,56; 256,128,112,128,64; 288,160,128,144,80; 320,192 160,160,96; 352,224,192,176,112; 384,256,224, 192,128; 416,320,256,224,144;  448,384,320,256,160];
                tmp = [tmp,tmp(:,5)];
                if HDR.MPEG.bitrate_index==0,
                        HDR.bitrate = NaN;
                elseif HDR.MPEG.bitrate_index==15,
                        fclose(HDR.FILE.FID);
                        fprintf(HDR.FILE.stderr,'SOPEN: corrupted MPEG file %s ',HDR.FileName);
                        return;
                else
                        HDR.bitrate = tmp(HDR.MPEG.bitrate_index,floor(HDR.VERSION)*3+HDR.MPEG.layer-3);
                end;
                
                switch HDR.MPEG.sampling_frequency_index,
                        case 0,
                                HDR.SampleRate = 44.100;
                        case 1,
                                HDR.SampleRate = 48.000;
                        case 2,
                                HDR.SampleRate = 32.000;
                        otherwise,
                                HDR.SampleRate = NaN;
                end;
                HDR.SampleRate_units = 'kHz';
                HDR.SampleRate = HDR.SampleRate*(2^(1-ceil(HDR.VERSION)));
                
                switch 4-HDR.MPEG.layer,	%Layer 
                        case 1,
                                HDR.SPR = 384;
                                slot = 32*HDR.MPEG.padding_bit; % bits, 4 bytes
                                HDR.FrameLengthInBytes = (12*HDR.bitrate/HDR.SampleRate+slot)*4; 
                        case {2,3},
                                HDR.SampleRate = 1152;
                                slot = 8*HDR.MPEG.padding_bit; % bits, 1 byte
                                HDR.FrameLengthInBytes = 144*HDR.bitrate/HDR.SampleRate+slot; 
                end;
                
                if ~HDR.MPEG.protection_bit,
                        HDR.MPEG.error_check = fread(HDR.FILE.FID,1,'uint16');
                end;
                
                HDR.MPEG.allocation = fread(HDR.FILE.FID,[1,32],'ubit4');
                HDR.MPEG.NoFB = sum(HDR.MPEG.allocation>0);
                HDR.MPEG.idx = find(HDR.MPEG.allocation>0);
                HDR.MPEG.scalefactor = fread(HDR.FILE.FID,[1,HDR.MPEG.NoFB],'ubit6');
                for k = HDR.MPEG.idx,
                        HDR.MPEG.temp(1:12,k) = fread(HDR.FILE.FID,[12,1],['ubit',int2str(HDR.MPEG.allocation(k))]);
                end;
                fprintf(HDR.FILE.stderr,'Warning SOPEN: MPEG not ready for use (%s)\n',HDR.FileName);
                HDR.FILE.OPEN = 1; 
        end;
        HDR.FILE.OPEN = 0; 
        fclose(HDR.FILE.FID);
        HDR.FILE.FID = -1; 
        return; 
        
        
elseif strcmp(HDR.TYPE,'QTFF'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-be');
        if ~isempty(findstr(PERMISSION,'r')),		%%%%% READ 
                HDR.FILE.OPEN = 1; 
                offset = 0; 
                while ~feof(HDR.FILE.FID),	
                        tagsize = fread(HDR.FILE.FID,1,'uint32');        % which size 
                        if ~isempty(tagsize),
                                offset = offset + tagsize; 
                                tag = setstr(fread(HDR.FILE.FID,[1,4],'char'));
                                if tagsize==0,
                                        tagsize=inf; %tagsize-8;        
                                elseif tagsize==1,
                                        tagsize=fread(HDR.FILE.FID,1,'uint64');        
                                end;
                                
                                if tagsize <= 8,
                                elseif strcmp(tag,'free'),
                                        val = fread(HDR.FILE.FID,[1,tagsize-8],'char');
                                        HDR.MOV.free = val;
                                elseif strcmp(tag,'skip'),
                                        val = fread(HDR.FILE.FID,[1,tagsize-8],'char');
                                        HDR.MOV.skip = val;
                                elseif strcmp(tag,'wide'),
                                        %val = fread(HDR.FILE.FID,[1,tagsize-8],'char');
                                        %HDR.MOV.wide = val;
                                elseif strcmp(tag,'pnot'),
                                        val = fread(HDR.FILE.FID,[1,tagsize-8],'char');
                                        HDR.MOV.pnot = val;
                                elseif strcmp(tag,'moov'),
                                        offset2 = 8;
                                        while offset2 < tagsize, 
                                                tagsize2 = fread(HDR.FILE.FID,1,'uint32');        % which size 
                                                if tagsize2==0,
                                                        tagsize2 = inf;
                                                elseif tagsize2==1,
                                                        tagsize2=fread(HDR.FILE.FID,1,'uint64');        
                                                end;
                                                offset2 = offset2 + tagsize2;                
                                                tag2 = setstr(fread(HDR.FILE.FID,[1,4],'char'));
                                                if tagsize2 <= 8,
                                                elseif strcmp(tag2,'mvhd'),
                                                        HDR.MOOV.Version = fread(HDR.FILE.FID,1,'char');
                                                        HDR.MOOV.Flags = fread(HDR.FILE.FID,3,'char');
                                                        HDR.MOOV.Times = fread(HDR.FILE.FID,5,'uint32');
                                                        HDR.T0 = datevec(HDR.MOOV.Times(1)/(3600*24))+[1904,0,0,0,0,0];
                                                        HDR.MOOV.prefVol = fread(HDR.FILE.FID,1,'uint16');
                                                        HDR.MOOV.reserved = fread(HDR.FILE.FID,10,'char');
                                                        HDR.MOOV.Matrix = fread(HDR.FILE.FID,[3,3],'int32')';
                                                        HDR.MOOV.Matrix(:,1:2) = HDR.MOOV.Matrix(:,1:2)/2^16; 
                                                        HDR.MOOV.Preview = fread(HDR.FILE.FID,5,'uint32');
                                                elseif strcmp(tag2,'trak'),
                                                        HDR.MOOV.trak = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                elseif strcmp(tag2,'cmov'),
                                                        HDR.MOOV.cmov = fread(HDR.FILE.FID,[1,tagsize2-8],'uchar');
                                                elseif strcmp(tag2,'free'),
                                                        HDR.MOOV.free = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                elseif strcmp(tag2,'clip'),
                                                        HDR.MOOV.clip = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                elseif strcmp(tag2,'udta'),
                                                        HDR.MOOV.udta = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                elseif strcmp(tag2,'ctab'),
                                                        HDR.MOOV.ctab = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                else
                                                end;
                                        end;
                                        %HDR.MOV.moov = fread(HDR.FILE.FID,[1,tagsize-8],'char');
                                        
                                elseif strcmp(tag,'mdat'),
                                        HDR.HeadLen = ftell(HDR.FILE.FID);        
                                        offset2 = 8;
                                        while offset2 < tagsize, 
                                                tagsize2 = fread(HDR.FILE.FID,1,'uint32');        % which size 
                                                tag2 = char(fread(HDR.FILE.FID,[1,4],'char'));
                                                if tagsize2==0,
                                                        tagsize2 = inf;
                                                elseif tagsize2==1,
                                                        tagsize2 = fread(HDR.FILE.FID,1,'uint64');        
                                                end;
                                                offset2  = offset2 + tagsize2;
                                                if tagsize2 <= 8,
                                                elseif strcmp(tag2,'mdat'),
                                                        HDR.MDAT.mdat = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                elseif strcmp(tag2,'wide'),
                                                        HDR.MDAT.wide = fread(HDR.FILE.FID,[1,tagsize2],'char');
                                                elseif strcmp(tag2,'clip'),
                                                        HDR.MDAT.clip = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                elseif strcmp(tag2,'udta'),
                                                        HDR.MDAT.udta = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                elseif strcmp(tag2,'ctab'),
                                                        HDR.MDAT.ctab = fread(HDR.FILE.FID,[1,tagsize2-8],'char');
                                                else
                                                end;
                                        end;
                                        %HDR.MOV.mdat = fread(HDR.FILE.FID,[1,tagsize-8],'char');
                                else
                                        val = fread(HDR.FILE.FID,[1,tagsize-8],'char');
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN Type=MOV: unknown Tag %s.\n',tag);
                                end;
                                fseek(HDR.FILE.FID,offset,'bof');
                        end;       
                end;
        end;        
        %fclose(HDR.FILE.FID);
        
        
elseif strcmp(HDR.TYPE,'ASF') ,
        if exist('asfopen','file'),
                HDR = asfopen(HDR,PERMISSION);
        else
                fprintf(1,'SOPEN ASF-File: Microsoft claims that its illegal to implement the ASF format.\n');
                fprintf(1,'     Anyway Microsoft provides the specification at http://www.microsoft.com/windows/windowsmedia/format/asfspec.aspx \n');
                fprintf(1,'     So, you can implement it and use it for your own purpose.\n');
        end; 
        
        
elseif strcmp(HDR.TYPE,'MIDI') | strcmp(HDR.TYPE,'RMID') ,
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,HDR.Endianity);

        if ~isempty(findstr(PERMISSION,'r')),		%%%%% READ 
                
                [tmp,c] = fread(HDR.FILE.FID,[1,4+12*strcmp(HDR.TYPE,'RMID') ],'char');
		tmp = char(tmp(c+(-3:0)));
                if ~strcmpi(tmp,'MThd'),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (MIDI): file %s might be corrupted 1\n',HDR.FileName);
                end;

                while ~feof(HDR.FILE.FID),	
                        tag     = setstr(tmp);
                        tagsize = fread(HDR.FILE.FID,1,'uint32');        % which size 
                        filepos = ftell(HDR.FILE.FID);
                        
                        if 0, 

			%%%% MIDI file format 	
                        elseif strcmpi(tag,'MThd');
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize/2],'uint16');
                                HDR.MIDI.Format = tmp(1);
                                HDR.NS = tmp(2);
				if tmp(3)<2^15,
		                        HDR.SampleRate = tmp(3);
				else	
					tmp4 = floor(tmp(3)/256);
					if tmp>127, 
						tmp4 = 256-tmp4; 
						HDR.SampleRate = (tmp4*rem(tmp(3),256));
					end
				end;
				CurrentTrack = 0; 

                        elseif strcmpi(tag,'MTrk');
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uint8');
				CurrentTrack = CurrentTrack + 1; 
				HDR.MIDI.Track{CurrentTrack} = tmp; 
				k = 1; 
				while 0,k<c,
					deltatime = 1; 
					while tmp(k)>127,
						deltatime = mod(tmp(k),128) + deltatime*128;
						k = k+1;
					end;
					deltatime = tmp(k) + deltatime*128;
					k = k+1;
					status_byte = tmp(k);
					k = k+1;
					
					if any(floor(status_byte/16)==[8:11]), % Channel Mode Message
						databyte = tmp(k:k+1);
						k = k+2;
				
					elseif any(floor(status_byte/16)==[12:14]), % Channel Voice Message
						databyte = tmp(k);
						k = k+1;

					elseif any(status_byte==hex2dec(['F0';'F7'])) % Sysex events
						len = 1; 
						while tmp(k)>127,
							len = mod(tmp(1),128) + len*128
							k = k+1;
						end;
						len = tmp(k) + len*128;
						data = tmp(k+(1:len));
				
					% System Common Messages 
					elseif status_byte==240, % F0 
					elseif status_byte==241, % F1 
						while tmp(k)<128,
							k = k+1;
						end;
					elseif status_byte==242, % F2 
						k = k + 1; 
					elseif status_byte==243, % F3 
						k = k + 1; 
					elseif status_byte==244, % F4 
					elseif status_byte==245, % F5 
					elseif status_byte==246, % F6 
					elseif status_byte==247, % F7 
					elseif status_byte==(248:254), % F7:FF 
								
					elseif (status_byte==255) % Meta Events
						type = tmp(k);
						k = k+1;
						len = 1; 
						while tmp(k)>127,
							len = mod(tmp(1),128) + len*128
							k = k+1;
						end;
						len = tmp(k) + len*128;
						data = tmp(k+1:min(k+len,length(tmp)));
						if 0, 
						elseif type==0,	HDR.MIDI.SequenceNumber = data;
						elseif type==1,	HDR.MIDI.TextEvent = char(data);
						elseif type==2,	HDR.Copyright = char(data);
						elseif type==3,	HDR.MIDI.SequenceTrackName = char(data);
						elseif type==4,	HDR.MIDI.InstrumentNumber = char(data);
						elseif type==5,	HDR.MIDI.Lyric = char(data);
						elseif type==6,	HDR.EVENT.POS = data;
						elseif type==7,	HDR.MIDI.CuePoint = char(data);
						elseif type==32,MDR.MIDI.ChannelPrefix = data;
						elseif type==47,MDR.MIDI.EndOfTrack = k;
						
						end;
					else
					end; 
				end;
                                
                        elseif ~isempty(tagsize)
                                fprintf(HDR.FILE.stderr,'Warning SOPEN (MIDI): unknown TAG in %s: %s(%i) \n',HDR.FileName,tag,tagsize);
                                [tmp,c] = fread(HDR.FILE.FID,[1,min(100,tagsize)],'uchar');
                                fprintf(HDR.FILE.stderr,'%s\n',char(tmp));
			end,

                        if ~isempty(tagsize)
                                status = fseek(HDR.FILE.FID,filepos+tagsize,'bof');
                                if status, 
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN (MIDI): fseek failed. Probably tagsize larger than end-of-file and/or file corrupted\n');
                                        fseek(HDR.FILE.FID,0,'eof');
                                end; 
                        end;
                        [tmp,c] = fread(HDR.FILE.FID,[1,4],'char');
		end;
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);

                HDR.FILE.POS = 0;
                HDR.FILE.OPEN = 1;
                HDR.NRec = 1;
        end; 
        
        
elseif strmatch(HDR.TYPE,['AIF';'IIF';'WAV';'AVI']),
	if strcmp(HDR.TYPE,'AIF')
	        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-be');
        else
		HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
	end;
	
        if ~isempty(findstr(PERMISSION,'r')),		%%%%% READ 

                tmp = setstr(fread(HDR.FILE.FID,[1,4],'char'));
                if ~strcmpi(tmp,'FORM') & ~strcmpi(tmp,'RIFF')
                        fprintf(HDR.FILE.stderr,'Warning SOPEN AIF/WAV-format: file %s might be corrupted 1\n',HDR.FileName);
                end;
                tagsize  = fread(HDR.FILE.FID,1,'uint32');        % which size
                tagsize0 = tagsize + rem(tagsize,2); 
                tmp = setstr(fread(HDR.FILE.FID,[1,4],'char'));
                if ~strncmpi(tmp,'AIF',3) & ~strncmpi(tmp,'WAVE',4) & ~strncmpi(tmp,'AVI ',4),
			% not (AIFF or AIFC or WAVE)
                        fprintf(HDR.FILE.stderr,'Warning SOPEN AIF/WAF-format: file %s might be corrupted 2\n',HDR.FileName);
                end;
                
                [tmp,c] = fread(HDR.FILE.FID,[1,4],'char');
                while ~feof(HDR.FILE.FID),	
                        tag     = setstr(tmp);
                        tagsize = fread(HDR.FILE.FID,1,'uint32');        % which size 
                        tagsize0= tagsize + rem(tagsize,2); 
                        filepos = ftell(HDR.FILE.FID);
                        
                        %%%% AIF - section %%%%%
                        if strcmpi(tag,'COMM')
                                if tagsize<18, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AIF: incorrect tag size\n');
                                        return;
                                end;
                                HDR.NS   = fread(HDR.FILE.FID,1,'uint16');
                                HDR.SPR  = fread(HDR.FILE.FID,1,'uint32');
                                HDR.AS.endpos = HDR.SPR;
                                HDR.Bits = fread(HDR.FILE.FID,1,'uint16');
                                %HDR.GDFTYP = ceil(HDR.Bits/8)*2-1; % unsigned integer of approbriate size;
                                if HDR.Bits == 8;
					HDR.GDFTYP = 'uint8';
                                elseif HDR.Bits == 16;
					HDR.GDFTYP = 'uint16';
                                elseif HDR.Bits == 32;
					HDR.GDFTYP = 'uint32';
                                else
					HDR.GDFTYP = ['ubit', int2str(HDR.Bits)];
				end;	
                                HDR.Cal  = 2^(1-HDR.Bits);
                                HDR.Off  = 0; 
                                HDR.AS.bpb = ceil(HDR.Bits/8)*HDR.NS;
                                
                                % HDR.SampleRate; % construct Extended 80bit IEEE 754 format 
                                tmp = fread(HDR.FILE.FID,1,'int16');
                                sgn = sign(tmp);
                                if tmp(1)>= 2^15; tmp(1)=tmp(1)-2^15; end;
                                e = tmp - 2^14 + 1;
                                tmp = fread(HDR.FILE.FID,2,'uint32');
                                HDR.SampleRate = sgn * (tmp(1)*(2^(e-31))+tmp(2)*2^(e-63));
                                HDR.Dur = HDR.SPR/HDR.SampleRate;
                                HDR.FILE.TYPE = 0;
                                
                                if tagsize>18,
                                        [tmp,c] = fread(HDR.FILE.FID,[1,4],'char');
                                        HDR.AIF.CompressionType = setstr(tmp);
                                        [tmp,c] = fread(HDR.FILE.FID,[1,tagsize-18-c],'char');
                                        HDR.AIF.CompressionName = tmp;
                                        
                                        if strcmpi(HDR.AIF.CompressionType,'NONE');
                                        elseif strcmpi(HDR.AIF.CompressionType,'fl32');
                                                HDR.GDFTYP = 'uint16';
                                                HDR.Cal = 1;
                                        elseif strcmpi(HDR.AIF.CompressionType,'fl64');
                                                HDR.GDFTYP = 'float64';
                                                HDR.Cal = 1;
                                        elseif strcmpi(HDR.AIF.CompressionType,'alaw');
                                                HDR.GDFTYP = 'uint8';
                                                HDR.AS.bpb = HDR.NS;
                                                %HDR.FILE.TYPE = 1;
                                                fprintf(HDR.FILE.stderr,'Warning SOPEN AIFC-format: data not scaled because of CompressionType ALAW\n');
                                                HDR.FLAG.UCAL = 1;
                                        elseif strcmpi(HDR.AIF.CompressionType,'ulaw');
                                                HDR.GDFTYP = 'uint8';
                                                HDR.AS.bpb = HDR.NS;
                                                HDR.FILE.TYPE = 1;  
                                                
                                                %%%% other compression types - currently not supported, probably obsolete
                                                %elseif strcmpi(HDR.AIF.CompressionType,'DWVW');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'GSM');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'ACE2');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'ACE8');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'ima4');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'MAC3');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'MAC6');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'Qclp');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'QDMC');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'rt24');
                                                %elseif strcmpi(HDR.AIF.CompressionType,'rt29');
                                        else
                                                fprintf(HDR.FILE.stderr,'Warning SOPEN AIFC-format: CompressionType %s is not supported\n', HDR.AIF.CompressionType);
                                        end;
                                end;	
                                
                        elseif strcmpi(tag,'SSND');
                                HDR.AIF.offset   = fread(HDR.FILE.FID,1,'int32');
                                HDR.AIF.blocksize= fread(HDR.FILE.FID,1,'int32');
                                HDR.AIF.SSND.tagsize = tagsize-8; 
				
                                HDR.HeadLen = filepos+8; 
                                %HDR.AIF.sounddata= fread(HDR.FILE.FID,tagsize-8,'uint8');
                                
                        elseif strcmpi(tag,'FVER');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN WAV: incorrect tag size\n');
                                        return;
                                end;
                                HDR.AIF.TimeStamp   = fread(HDR.FILE.FID,1,'uint32');
                                
                        elseif strcmp(tag,'DATA') & strcmp(HDR.TYPE,'AIF') ;	% AIF uses upper case, there is a potential conflict with WAV using lower case data  
                                HDR.AIF.DATA  = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                
                        elseif strcmpi(tag,'INST');   % not sure if this is ok !
                                %HDR.AIF.INST  = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                %HDR.AIF.INST.notes  = fread(HDR.FILE.FID,[1,6],'char');
                                HDR.AIF.INST.baseNote  = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.detune    = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.lowNote   = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.highNote  = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.lowvelocity  = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.highvelocity = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.gain      = fread(HDR.FILE.FID,1,'int16');
                                
                                HDR.AIF.INST.sustainLoop_PlayMode = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.sustainLoop = fread(HDR.FILE.FID,2,'uint16');
                                HDR.AIF.INST.releaseLoop_PlayMode = fread(HDR.FILE.FID,1,'char');
                                HDR.AIF.INST.releaseLoop = fread(HDR.FILE.FID,2,'uint16');
                                
                        elseif strcmpi(tag,'MIDI');
                                HDR.AIF.MIDI = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                
                        elseif strcmpi(tag,'AESD');
                                HDR.AIF.AESD = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                
                        elseif strcmpi(tag,'APPL');
                                HDR.AIF.APPL = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                
                        elseif strcmpi(tag,'COMT');
                                HDR.AIF.COMT = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                
                        elseif strcmpi(tag,'ANNO');
                                HDR.AIF.ANNO = setstr(fread(HDR.FILE.FID,[1,tagsize],'uchar'));
                                
                        elseif strcmpi(tag,'(c) ');
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.Copyright = setstr(tmp);
                                
                                %%%% WAV - section %%%%%
                        elseif strcmpi(tag,'fmt ')
                                if tagsize<14, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN WAV: incorrect tag size\n');
                                        return;
                                end;
                                HDR.WAV.Format = fread(HDR.FILE.FID,1,'uint16');
                                HDR.NS = fread(HDR.FILE.FID,1,'uint16');
                                HDR.SampleRate = fread(HDR.FILE.FID,1,'uint32');
                                HDR.WAV.AvgBytesPerSec = fread(HDR.FILE.FID,1,'uint32');
                                HDR.WAV.BlockAlign = fread(HDR.FILE.FID,1,'uint16');
                                if HDR.WAV.Format==1,	% PCM format
                                        HDR.Bits = fread(HDR.FILE.FID,1,'uint16');
                                        HDR.Off = 0;
                                        HDR.Cal = 2^(1-8*ceil(HDR.Bits/8));
                                        if HDR.Bits<=8,
                                                HDR.GDFTYP = 'uchar'; 
                                                HDR.Off =  1;
                                                %HDR.Cal = HDR.Cal*2;
                                        elseif HDR.Bits<=16,
                                                HDR.GDFTYP = 'int16'; 
                                        elseif HDR.Bits<=24,
                                                HDR.GDFTYP = 'bit24'; 
                                        elseif HDR.Bits<=32,
                                                HDR.GDFTYP = 'int32'; 
                                        end;
                                else 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN WAV: format type %i not supported\n',HDR.WAV.Format);	
                                        fclose(HDR.FILE.FID);
                                        return; 
                                end;
                                if tagsize>16,
                                        HDR.WAV.cbSize = fread(HDR.FILE.FID,1,'uint16');
                                end;
                                
                        elseif strcmp(tag,'data') & strcmp(HDR.TYPE,'WAV') ;	% AIF uses upper case, there is a potential conflict with WAV using lower case data  
                                HDR.HeadLen = filepos; 
                                if HDR.WAV.Format == 1, 
                                        HDR.AS.bpb = HDR.NS * ceil(HDR.Bits/8);
                                        HDR.SPR = tagsize/HDR.AS.bpb;
                                        HDR.Dur = HDR.SPR/HDR.SampleRate;
                                        HDR.AS.endpos = HDR.SPR;
                                        
                                else 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN WAV: format type %i not supported\n',HDR.WAV.Format);	
                                end;
                                
                        elseif strcmpi(tag,'fact');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN WAV: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.FACT = setstr(tmp);
                                
                        elseif strcmpi(tag,'disp');
                                if tagsize<8, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN WAV: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.DISP = setstr(tmp);
                                if ~all(tmp(1:8)==[0,1,0,0,0,0,1,1])
                                        HDR.RIFF.DISPTEXT = setstr(tmp(5:length(tmp)));
                                end;
                                
                        elseif strcmpi(tag,'list');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN WAV: incorrect tag size\n');
                                        return;
                                end;
                                
                                if ~isfield(HDR,'RIFF');
                                        HDR.RIFF.N1 = 1;
                                elseif ~isfield(HDR.RIFF,'N');
                                        HDR.RIFF.N1 = 1;
                                else
                                        HDR.RIFF.N1 = HDR.RIFF.N1+1;
                                end;
                                
                                %HDR.RIFF.list = setstr(tmp);
                                [tag,c1]  = fread(HDR.FILE.FID,[1,4],'char');
				tag = char(tag);
                                [val,c2]  = fread(HDR.FILE.FID,[1,tagsize-4],'char');
				HDR.RIFF = setfield(HDR.RIFF,tag,char(val));
                                if 1,
				elseif strcmp(tag,'INFO'),
                                        HDR.RIFF.INFO=val;
                                elseif strcmp(tag,'movi'),
                                        HDR.RIFF.movi = val;
                                elseif strcmp(tag,'hdrl'),
                                        HDR.RIFF.hdr1 = val;
					
                                elseif 0,strcmp(tag,'mdat'),
                                        %HDR.RIFF.mdat = val;
                                else
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN Type=RIFF: unknown Tag %s.\n',tag);
                                end;
                                % AVI  audio video interleave format 	
                        elseif strcmpi(tag,'movi');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AVI: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.movi = setstr(tmp);
                                
                        elseif strcmp(tag,'idx1');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AVI: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.idx1 = setstr(tmp);
                                
                        elseif strcmpi(tag,'junk');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AVI: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.junk = setstr(tmp);
                                
                        elseif strcmpi(tag,'MARK');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AVI: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.MARK = setstr(tmp);
                                
                        elseif strcmpi(tag,'AUTH');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AVI: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.AUTH = setstr(tmp);
                                
                        elseif strcmpi(tag,'NAME');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AVI: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.NAME = setstr(tmp);
                                
                        elseif strcmpi(tag,'afsp');
                                if tagsize<4, 
                                        fprintf(HDR.FILE.stderr,'Error SOPEN AVI: incorrect tag size\n');
                                        return;
                                end;
                                [tmp,c] = fread(HDR.FILE.FID,[1,tagsize],'uchar');
                                HDR.RIFF.afsp = setstr(tmp);
                                
                        elseif ~isempty(tagsize)
                                fprintf(HDR.FILE.stderr,'Warning SOPEN AIF/WAV: unknown TAG in %s: %s(%i) \n',HDR.FileName,tag,tagsize);
                                [tmp,c] = fread(HDR.FILE.FID,[1,min(100,tagsize)],'uchar');
                                fprintf(HDR.FILE.stderr,'%s\n',char(tmp));
                        end;

                        if ~isempty(tagsize)
                                status = fseek(HDR.FILE.FID,filepos+tagsize0,'bof');
                                if status, 
                                        fprintf(HDR.FILE.stderr,'Warning SOPEN (WAF/AIF/AVI): fseek failed. Probably tagsize larger than end-of-file and/or file corrupted\n');
                                        fseek(HDR.FILE.FID,0,'eof');
                                end; 
                        end;
                        [tmp,c] = fread(HDR.FILE.FID,[1,4],'char');
                end;
                
		if strncmpi(tmp,'AIF',3),
                        if HDR.AIF.SSND.tagsize~=HDR.SPR*HDR.AS.bpb,
                                fprintf(HDR.FILE.stderr,'Warning SOPEN AIF: Number of samples do not fit %i vs %i\n',tmp,HDR.SPR);
                        end;
                end;
		                
                if ~isfield(HDR,'HeadLen') 
                        fprintf(HDR.FILE.stderr,'Warning SOPEN AIF/WAV: missing data section\n');
                else
                        status = fseek(HDR.FILE.FID, HDR.HeadLen, 'bof');
                end;
		
		if isnan(HDR.NS), return; end; 
	    	[d,l,d1,b,HDR.GDFTYP] = gdfdatatype(HDR.GDFTYP);

                % define Calib: implements S = (S+.5)*HDR.Cal - HDR.Off;
                HDR.Calib = [repmat(.5,1,HDR.NS);eye(HDR.NS)] * diag(repmat(HDR.Cal,1,HDR.NS));
                HDR.Calib(1,:) = HDR.Calib(1,:) - HDR.Off;
		HDR.Label = repmat(' ',HDR.NS,1);

                HDR.FILE.POS = 0;
                HDR.FILE.OPEN = 1;
                HDR.NRec = 1;
                
                
        elseif ~isempty(findstr(PERMISSION,'w')),	%%%%% WRITE 
                HDR.FILE.OPEN = 3; 
                if strcmp(HDR.TYPE,'AIF') 
                        fwrite(HDR.FILE.FID,'FORM','char');	
                        fwrite(HDR.FILE.FID,0,'uint32');	
                        fwrite(HDR.FILE.FID,'AIFFCOMM','char');	
                        fwrite(HDR.FILE.FID,18,'uint32');	
                        fwrite(HDR.FILE.FID,HDR.NS,'uint16');	
                        fwrite(HDR.FILE.FID,HDR.SPR,'uint32');	
                        fwrite(HDR.FILE.FID,HDR.Bits,'uint16');	
                        
                        %HDR.GDFTYP = ceil(HDR.Bits/8)*2-1; % unsigned integer of appropriate size;
                        HDR.GDFTYP = ['ubit', int2str(HDR.Bits)];
                        HDR.Cal    = 2^(1-HDR.Bits);
                        HDR.AS.bpb = ceil(HDR.Bits/8)*HDR.NS;
                        
                        [f,e] = log2(HDR.SampleRate);
                        tmp = e + 2^14 - 1;
                        if tmp<0, tmp = tmp + 2^15; end;
                        fwrite(HDR.FILE.FID,tmp,'uint16');	
                        fwrite(HDR.FILE.FID,[bitshift(abs(f),31),bitshift(abs(f),63)],'uint32');	
                        
                        HDR.AS.bpb = HDR.NS * ceil(HDR.Bits/8);
                        tagsize = HDR.SPR*HDR.AS.bpb + 8;
                        HDR.Dur = HDR.SPR/HDR.SampleRate;
                        HDR.AS.endpos = HDR.SPR;
                        
                        if 0; isfield(HDR.AIF,'INST');	% does not work yet
                                fwrite(HDR.FILE.FID,'SSND','char');	
                                fwrite(HDR.FILE.FID,20,'uint32');	
                                
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.baseNote,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.detune,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.lowNote,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.highNote,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.lowvelocity,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.highvelocity,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.gain,'int16');
                                
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.sustainLoop_PlayMode,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.sustainLoop,'uint16');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.releaseLoop_PlayMode,'char');
                                fwrite(HDR.FILE.FID,HDR.AIF.INST.releaseLoop,'uint16');
                        end;
                        
                        fwrite(HDR.FILE.FID,'SSND','char');	
                        HDR.WAV.posis = [4, ftell(HDR.FILE.FID)];
                        fwrite(HDR.FILE.FID,[tagsize,0,0],'uint32');	
                        
                        HDR.HeadLen = ftell(HDR.FILE.FID);
                        
                elseif  strcmp(HDR.TYPE,'WAV'),
                        fwrite(HDR.FILE.FID,'RIFF','char');	
                        fwrite(HDR.FILE.FID,0,'uint32');	
                        fwrite(HDR.FILE.FID,'WAVEfmt ','char');	
                        fwrite(HDR.FILE.FID,16,'uint32');	
                        fwrite(HDR.FILE.FID,[1,HDR.NS],'uint16');	
                        fwrite(HDR.FILE.FID,[HDR.SampleRate,HDR.Bits/8*HDR.NS*HDR.SampleRate],'uint32');	
                        fwrite(HDR.FILE.FID,[HDR.Bits/8*HDR.NS,HDR.Bits],'uint16');	
                        
                        if isfield(HDR,'Copyright'),
                                fwrite(HDR.FILE.FID,'(c) ','char');	
                                if rem(length(HDR.Copyright),2),
                                        HDR.Copyright(length(HDR.Copyright)+1)=' ';
                                end;	
                                fwrite(HDR.FILE.FID,length(HDR.Copyright),'uint32');	
                                fwrite(HDR.FILE.FID,HDR.Copyright,'char');	
                        end;
                        
                        HDR.Off = 0;
                        HDR.Cal = 2^(1-8*ceil(HDR.Bits/8));
                        if HDR.Bits<=8,
                                HDR.GDFTYP = 'uchar'; 
                                HDR.Off =  1;
                                %HDR.Cal = HDR.Cal*2;
                        elseif HDR.Bits<=16,
                                HDR.GDFTYP = 'int16'; 
                        elseif HDR.Bits<=24,
                                HDR.GDFTYP = 'bit24'; 
                        elseif HDR.Bits<=32,
                                HDR.GDFTYP = 'int32'; 
                        end;
                        
                        HDR.AS.bpb = HDR.NS * ceil(HDR.Bits/8);
                        tagsize = HDR.SPR*HDR.AS.bpb;
                        HDR.Dur = HDR.SPR/HDR.SampleRate;
                        HDR.AS.endpos = HDR.SPR;
                        
                        fwrite(HDR.FILE.FID,'data','char');	
                        HDR.WAV.posis=[4,ftell(HDR.FILE.FID)];
                        fwrite(HDR.FILE.FID,tagsize,'uint32');	
                        
                        if rem(tagsize,2)
                                fprintf(HDR.FILE.stderr,'Error SOPEN WAV: data section has odd number of samples.\n. This violates the WAV specification\n');
                                fclose(HDR.FILE.FID);
                                HDR.FILE.OPEN = 0;
                                return;  
                        end;
                        
                        HDR.HeadLen = ftell(HDR.FILE.FID);
                end;
        end;

        
elseif strcmp(HDR.TYPE,'FLAC'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-be');
        
	if HDR.FILE.FID > 0,
    	        HDR.magic  = fread(HDR.FILE.FID,[1,4],'char');
		
		% read METADATA_BLOCK
		% 	read METADATA_BLOCK_HEADER
    	        tmp = fread(HDR.FILE.FID,[1,4],'uchar')
		while (tmp(1)<128),
			BLOCK_TYPE = mod(tmp(1),128);
			LEN = tmp(2:4)*2.^[0;8;16];
			POS = ftell(HDR.FILE.FID);
			if (BLOCK_TYPE == 0),		% STREAMINFO
				minblksz = fread(HDR.FILE.FID,1,'uint16')
				maxblksz = fread(HDR.FILE.FID,1,'uint16')
				minfrmsz = 2.^[0,8,16]*fread(HDR.FILE.FID,3,'uint8')
				maxfrmsz = 2.^[0,8,16]*fread(HDR.FILE.FID,3,'uint8')
				%Fs = fread(HDR.FILE.FID,3,'ubit20')
			elseif (BLOCK_TYPE == 1),	% PADDING
			elseif (BLOCK_TYPE == 2),	% APPLICATION
				HDR.FLAC.Reg.Appl.ID = fread(HDR.FILE.FID,1,'uint32')
			elseif (BLOCK_TYPE == 3),	% SEEKTABLE
				HDR.EVENT.N = LEN/18;
				for k = 1:LEN/18,
	    				HDR.EVENT.POS(k) = 2.^[0,32]*fread(HDR.FILE.FID,2,'uint32');
	    				HDR.EVENT.DUR(k) = 2.^[0,32]*fread(HDR.FILE.FID,2,'uint32');
	    				HDR.EVENT.nos(k) = fread(HDR.FILE.FID,1,'uint16');
				
				end;
			elseif (BLOCK_TYPE == 4),	% VORBIS_COMMENT
			elseif (BLOCK_TYPE == 5),	% CUESHEET
			else					% reserved
			end;
			
			fseek(HDR.FILE.FID, POS+LEN,'bof');
        	        tmp = fread(HDR.FILE.FID,[1,4],'uchar')
		end;
		
		% 	read METADATA_BLOCK_DATA

		% read METADATA_BLOCK_DATA
		% 	read METADATA_BLOCK_STREAMINFO
		% 	read METADATA_BLOCK_PADDING
		% 	read METADATA_BLOCK_APPLICATION
		% 	read METADATA_BLOCK_SEEKTABLE
		% 	read METADATA_BLOCK_COMMENT
		% 	read METADATA_BLOCK_CUESHEET

		% read FRAME
		%	read FRAME_HEADER
		%	read FRAME_SUBFRAME
		%		read FRAME_SUBFRAME_HEADER
		%	read FRAME_HEADER

                fclose(HDR.FILE.FID)        

                fprintf(HDR.FILE.stderr,'Warning SOPEN: FLAC not ready for use\n');
		return;
        end;

        
elseif strcmp(HDR.TYPE,'OGG'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        
	if HDR.FILE.FID > 0,
		% chunk header
		tmp = fread(HDR.FILE.FID,1,'uchar');
		QualityIndex = mod(tmp(1),64);
		if (tmp(1)<128), % golden frame 
			tmp = fread(HDR.FILE.FID,2,'uchar');
			HDR.VERSION = tmp(1);
			HDR.VP3.Version = floor(tmp(2)/8);
			HDR.OGG.KeyFrameCodingMethod = floor(mod(tmp(2),8)/4);
		end;
		
		% block coding information
		% coding mode info
		% motion vectors
		% DC coefficients
		% DC coefficients
		% 1st AC coefficients
		% 2nd AC coefficients
		% ...
		% 63rd AC coefficient

                fclose(HDR.FILE.FID);        
                fprintf(HDR.FILE.stderr,'Warning SOPEN: OGG not ready for use\n');
		return;
        end;

        
elseif strcmp(HDR.TYPE,'RMF'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        
	if HDR.FILE.FID > 0,
                fclose(HDR.FILE.FID)        

                fprintf(HDR.FILE.stderr,'Warning SOPEN: RMF not ready for use\n');
		return;
        end;

        
elseif strcmp(HDR.TYPE,'EGI'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-be');
        
        HDR.VERSION  = fread(HDR.FILE.FID,1,'uint32');
        
        if ~(HDR.VERSION >= 2 & HDR.VERSION <= 7),
                %   fprintf(HDR.FILE.stderr,'EGI Simple Binary Versions 2-7 supported only.\n');
        end;
        
        HDR.T0 = fread(HDR.FILE.FID,[1,6],'uint16');
        millisecond = fread(HDR.FILE.FID,1,'uint32');
        HDR.T0(6) = HDR.T0(6) + millisecond/1000;
        
        HDR.SampleRate = fread(HDR.FILE.FID,1,'uint16');
        HDR.NS   = fread(HDR.FILE.FID,1,'uint16');
        HDR.gain = fread(HDR.FILE.FID,1,'uint16');
        HDR.Bits = fread(HDR.FILE.FID,1,'uint16');
        HDR.DigMax  = 2^HDR.Bits;
        HDR.PhysMax = fread(HDR.FILE.FID,1,'uint16');
        if ( HDR.Bits ~= 0 & HDR.PhysMax ~= 0 )
                HDR.Cal = (HDR.PhysMax/HDR.DigMax);
        else
                HDR.Cal = 1;
        end;
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal,HDR.NS+1,HDR.NS);
        HDR.PhysDim = 'uV';
        HDR.Label = char(zeros(HDR.NS,5));
        for k=1:HDR.NS,
                HDR.Label(k,:)=sprintf('# %3i',k);
        end;
        
        HDR.categories = 0;
        HDR.EGI.catname= {};
        
        if any(HDR.VERSION==[2,4,6]),
                HDR.SPR  = fread(HDR.FILE.FID, 1 ,'int32');
                HDR.EVENT.N = fread(HDR.FILE.FID,1,'int16');
                HDR.NRec = 1;
                HDR.FLAG.TRIGGERED = logical(0); 
                HDR.AS.spb = HDR.NS;
                HDR.AS.endpos = HDR.SPR;
                HDR.Dur = 1/HDR.SampleRate;
        elseif any(HDR.VERSION==[3,5,7]),
                HDR.EGI.categories = fread(HDR.FILE.FID,1,'uint16');
                if (HDR.EGI.categories),
                        for i=1:HDR.EGI.categories,
                                catname_len(i) = fread(HDR.FILE.FID,1,'uchar');
                                HDR.EGI.catname{i} = char(fread(HDR.FILE.FID,catname_len(i),'uchar'))';
                        end
                end
                HDR.NRec = fread(HDR.FILE.FID,1,'int16');
                HDR.SPR  = fread(HDR.FILE.FID,1,'int32');
                HDR.EVENT.N = fread(HDR.FILE.FID,1,'int16');
                HDR.FLAG.TRIGGERED = logical(1); 
                HDR.AS.spb = HDR.SPR*(HDR.NS+HDR.EVENT.N);
                HDR.AS.endpos = HDR.NRec;
                HDR.Dur = HDR.SPR/HDR.SampleRate;
        else
                fprintf(HDR.FILE.stderr,'Invalid EGI version %i\n',HDR.VERSION);
                return;
        end
        
        % get datatype from version number
        if any(HDR.VERSION==[2,3]),
                HDR.GDFTYP = 'int16';
                HDR.AS.bpb = HDR.AS.spb*2;
        elseif any(HDR.VERSION==[4,5]),
                HDR.GDFTYP = 'float32';
                HDR.AS.bpb = HDR.AS.spb*4;
        elseif any(HDR.VERSION==[6,7]),
                HDR.GDFTYP = 'float64';
                HDR.AS.bpb = HDR.AS.spb*8;
        else
                error('Unknown data format');
        end
        HDR.AS.bpb = HDR.AS.bpb + 6*HDR.FLAG.TRIGGERED;
        
        tmp = fread(HDR.FILE.FID,[4,HDR.EVENT.N],'uchar');
        HDR.EGI.eventcode = reshape(tmp,[4,HDR.EVENT.N])';
        HDR.EVENT.TYP = HDR.EGI.eventcode*(2.^[24;16;8;0]);
        
        HDR.HeadLen = ftell(HDR.FILE.FID);
        HDR.FILE.POS= 0;
	HDR.FILE.OPEN = 1; 


elseif strcmp(HDR.TYPE,'TEAM'),		% Nicolet TEAM file format
        % implementation of this format is not finished yet.
        
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        %%%%% X-Header %%%%%
        HDR.VERSION = fread(HDR.FILE.FID,1,'int16');
        HDR.NS = fread(HDR.FILE.FID,1,'int16');
        HDR.NRec = fread(HDR.FILE.FID,1,'int16');
        HDR.TEAM.Length = fread(HDR.FILE.FID,1,'int32');
        HDR.TEAM.NSKIP = fread(HDR.FILE.FID,1,'int32');
        HDR.SPR = fread(HDR.FILE.FID,1,'int32');
        HDR.Samptype = fread(HDR.FILE.FID,1,'int16');
        if   	HDR.Samptype==2, HDR.GDFTYP = 'int16';
        elseif 	HDR.Samptype==4, HDR.GDFTYP = 'float32'; 
        else
                fprintf(HDR.FILE.stderr,'Error SOPEN TEAM-format: invalid file\n');
                fclose(HDR.FILE.FID);
                return;
        end;	
        HDR.XLabel = fread(HDR.FILE.FID,[1,8],'char');
        HDR.X0 = fread(HDR.FILE.FID,1,'float');
        HDR.TEAM.Xstep = fread(HDR.FILE.FID,1,'float');
        HDR.SampleRate = 1/HDR.TEAM.Xstep;
        tmp = fread(HDR.FILE.FID,[1,6],'uchar');
        tmp(1) = tmp(1) + 1980;
        HDR.T0 = tmp([4,5,6,1,2,3]);
        
        HDR.EVENT.N   = fread(HDR.FILE.FID,1,'int16');
        HDR.TEAM.Nsegments = fread(HDR.FILE.FID,1,'int16');
        HDR.TEAM.SegmentOffset = fread(HDR.FILE.FID,1,'int32');
        HDR.XPhysDim = fread(HDR.FILE.FID,[1,8],'char');
        HDR.TEAM.RecInfoOffset = fread(HDR.FILE.FID,1,'int32');
        status = fseek(HDR.FILE.FID,256,'bof');
        %%%%% Y-Header %%%%%
        for k = 1:HDR.NS,
                HDR.Label(k,1:7) = fread(HDR.FILE.FID,[1,7],'char');
                HDR.PhysDim(k,1:7) = fread(HDR.FILE.FID,[1,7],'char');
                HDR.Off(1,k) = fread(HDR.FILE.FID,1,'float');
                HDR.Cal(1,k) = fread(HDR.FILE.FID,1,'float');
                HDR.PhysMax(1,k) = fread(HDR.FILE.FID,1,'float');
                HDR.PhysMin(1,k) = fread(HDR.FILE.FID,1,'float');
                status = fseek(HDR.FILE.FID,2,'cof');
        end;
        HDR.HeadLen = 256+HDR.NS*32;
        
        % Digital (event) information 
        HDR.TEAM.DigitalOffset = 256 + 32*HDR.NS + HDR.NS*HDR.NRec*HDR.SPR*HDR.Samptype;
        status = fseek(HDR.FILE.FID,HDR.TEAM.DigitalOffset,'bof');
        if HDR.TEAM.DigitalOffset < HDR.TEAM.SegmentOffset,
                HDR.EventLabels = setstr(fread(HDR.FILE.FID,[16,HDR.EVENT.N],'char')');
                
                % Events could be detected in this way
                % HDR.Events = zeros(HDR.SPR*HDR.NRec,1);
                % for k = 1:ceil(HDR.EVENT.N/16)
                %	HDR.Events = HDR.Events + 2^(16*k-16)*fread(HDR.FILE.FID,HDR.SPR*HDR.NRec,'uint16');
                % end;
        end;
        
        % Segment information block entries 
        if HDR.TEAM.Nsegments,
                fseek(HDR.FILE.FID,HDR.TEAM.SegmentOffset,'bof');
                for k = 1:HDR.TEAM.Nsegments, 
                        HDR.TEAM.NSKIP(k) = fread(HDR.FILE.FID,1,'int32');
                        HDR.SPR(k)  = fread(HDR.FILE.FID,1,'int32');
                        HDR.X0(k) = fread(HDR.FILE.FID,1,'float');
                        HDR.Xstep(k) = fread(HDR.FILE.FID,1,'float');
                        status = fseek(HDR.FILE.FID,8,'cof');
                end;
        end;
        
        % Recording information block entries
        if HDR.TEAM.RecInfoOffset,
                status = fseek(HDR.FILE.FID,HDR.TEAM.RecInfoOffset,'bof');
                blockinformation = fread(HDR.FILE.FID,[1,32],'char');
                for k = 1:HDR.NRec, 
                        HDR.TRIGGER.Time(k) = fread(HDR.FILE.FID,1,'double');
                        HDR.TRIGGER.Date(k,1:3) = fread(HDR.FILE.FID,[1,3],'uint8');
                        fseek(HDR.FILE.FID,20,'cof');
                end;
                HDR.TRIGGER.Date(k,1) = HDR.TRIGGER.Date(k,1) + 1900;
        end;
        fprintf(HDR.FILE.stderr,'Warning SOPEN: Implementing Nicolet TEAM file format not completed yet. Contact <a.schloegl@ieee.org> if you are interested in this feature.\n');
        fclose(HDR.FILE.FID);
        
        
elseif strcmp(HDR.TYPE,'WFT'),	% implementation of this format is not finished yet.
        
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        [s,c] = fread(HDR.FILE.FID,1536,'char');
        [tmp,s] = strtok(s,setstr([0,32]));
        Nic_id0 = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        Niv_id1 = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        Nic_id2 = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        User_id = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.HeadLen = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.FILE.Size = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.VERSION = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.WFT.WaveformTitle = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.T0(1) = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.T0(1,2) = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.T0(1,3) = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        tmp = str2double(tmp);
        HDR.T0(1,4:6) = [floor(tmp/3600000),floor(rem(tmp,3600000)/60000),rem(tmp,60000)];
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.SPR = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.Off = str2double(tmp);
        [tmp,s] = strtok(s,setstr([0,32]));
        HDR.Cal = str2double(tmp);
        
        fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
        
        fprintf(HDR.FILE.stderr,'Warning SOPEN: Implementing Nicolet WFT file format not completed yet. Contact <a.schloegl@ieee.org> if you are interested in this feature.\n');
        fclose(HDR.FILE.FID);
        
        
elseif strcmp(HDR.TYPE,'WG1'),

        if ~isempty(findstr(PERMISSION,'r')),		%%%%% READ 
                HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,HDR.Endianity);
                
                HDR.VERSION = dec2hex(fread(HDR.FILE.FID,1,'uint32')); 
                HDR.WG1.MachineId = fread(HDR.FILE.FID,1,'uint32');
                HDR.WG1.Day = fread(HDR.FILE.FID,1,'uint32'); 
                HDR.WG1.millisec = fread(HDR.FILE.FID,1,'uint32');
		HDR.T0    = datevec(HDR.WG1.Day-15755-hex2dec('250000'));
		HDR.T0(1) = HDR.T0(1) + 1970;
		HDR.T0(4) = floor(HDR.WG1.millisec/3600000);
		HDR.T0(5) = mod(floor(HDR.WG1.millisec/60000),60);
		HDR.T0(6) = mod(HDR.WG1.millisec/1000,60);
                dT = fread(HDR.FILE.FID,1,'uint32');
                HDR.SampleRate = 1e6/dT;
                HDR.WG1.pdata = fread(HDR.FILE.FID,1,'uint16');
                HDR.NS = fread(HDR.FILE.FID,1,'uint16'); 
                HDR.WG1.poffset = fread(HDR.FILE.FID,1,'uint16');
                HDR.WG1.pad1 = fread(HDR.FILE.FID,38,'char');
		HDR.Cal = repmat(NaN,HDR.NS,1);
		HDR.ChanSelect = repmat(NaN,HDR.NS,1);
                for k=1:HDR.NS,
                        Label(k,1:8) = fread(HDR.FILE.FID,[1,8],'char');
                        HDR.Cal(k,1) = fread(HDR.FILE.FID,1,'uint32')/1000;
                        tmp = fread(HDR.FILE.FID,[1,2],'uint16');
                        HDR.ChanSelect(k) = tmp(1)+1;
                end;
		HDR.Label = char(Label);
		HDR.Calib = sparse(2:HDR.NS+1,HDR.ChanSelect,HDR.Cal);

                status = fseek(HDR.FILE.FID,7*256,'bof');
                HDR.WG1.neco1 = fread(HDR.FILE.FID,1,'uint32');
                HDR.Patient.Id = fread(HDR.FILE.FID,[1,12],'char');
                HDR.Patient.LastName = fread(HDR.FILE.FID,[1,20],'char');
                HDR.Patient.text1 = fread(HDR.FILE.FID,[1,20],'char');
                HDR.Patient.FirstName = fread(HDR.FILE.FID,[1,20],'char');
                HDR.Patient.Sex = fread(HDR.FILE.FID,[1,2],'char');
                HDR.Patient.vata = fread(HDR.FILE.FID,[1,8],'char');
                HDR.Patient.text2 = fread(HDR.FILE.FID,[1,14],'char');
                HDR.WG1.Datum = fread(HDR.FILE.FID,1,'uint32');
                HDR.WG1.mstime = fread(HDR.FILE.FID,1,'uint32');
                HDR.WG1.nic = fread(HDR.FILE.FID,[1,4],'uint32');
                HDR.WG1.neco3 = fread(HDR.FILE.FID,1,'uint32');
               
                status = fseek(HDR.FILE.FID,128,'cof');
                HDR.HeadLen = ftell(HDR.FILE.FID);
                HDR.FILE.OPEN = 1; 
		HDR.FILE.POS  = 0; 

		HDR.WG1.szBlock  = 256;  
		HDR.WG1.szOffset = 128;
		HDR.WG1.szExtra  = HDR.WG1.pdata-(HDR.NS+HDR.WG1.poffset);
		szOneRec = HDR.WG1.szOffset*4+(HDR.NS+HDR.WG1.szExtra)*HDR.WG1.szBlock;
		HDR.AS.bpb = szOneRec;
		HDR.WG1.szRecs = floor((HDR.FILE.size-HDR.HeadLen)/HDR.AS.bpb);
		HDR.WG1.szData = HDR.WG1.szBlock*HDR.WG1.szRecs;
    		HDR.WG1.unknownNr = 11;
        	conv = round(19*sinh((0:127)/19));
		conv = [conv, HDR.WG1.unknownNr, -conv(end:-1:2)];
    		HDR.WG1.conv = conv;

		HDR.NRec = HDR.WG1.szRecs;
		HDR.SPR  = HDR.WG1.szBlock;
		HDR.Dur  = HDR.SPR/HDR.SampleRate;
		HDR.AS.endpos = HDR.NRec*HDR.SPR;
		
		%----- load event information -----
		eventFile = fullfile(HDR.FILE.Path,[HDR.FILE.Name, '.wg2']);
		if ~exist(eventFile,'file')
			eventFile = fullfile(HDR.FILE.Path,[HDR.FILE.Name, '.WG2']);
		end;	
		if exist(eventFile,'file')
    			fid= fopen(eventFile,'r');
    			nr = 1;
			[s,c] = fread(fid,1,'uint32');
    			while ~feof(fid)
        			HDR.EVENT.POS(nr,1) = s;
        			pad = fread(fid,3,'uint32');
        			len = fread(fid,1,'uint8');
        			tmp = char(fread(fid,[1,47], 'char'));
				HDR.EVENT.Desc{nr,1} = tmp(1:len);  
    				% find string between quotation marks
				%  HDR.EVENT.Desc{nr}=regexpi(Event,'(?<=\'').*(?=\'')','match','once');
				[s,c] = fread(fid,1,'uint32');
        			nr  = nr+1;
		        end;
    			HDR.EVENT.TYP = zeros(size(HDR.EVENT.POS));
			fclose(fid);
		end;
        end;

        
elseif strcmp(HDR.TYPE,'LDR'),
        HDR = openldr(HDR,PERMISSION);      
        
        
elseif strcmp(HDR.TYPE,'SMA'),  % under constructions
        PERMISSION = PERMISSION(PERMISSION~='b');
        try     % MatLAB default is binary, force Mode='rt';
                HDR.FILE.FID = fopen(HDR.FileName,[PERMISSION,'t'],'ieee-le');
        catch 	% Octave 2.1.50 default is text, but does not support Mode='rt', 
                HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        end
        numbegin=0;
        HDR.H1 = '';
        delim = char([abs('"='),10,13]);
        while ~numbegin,
                line = fgetl(HDR.FILE.FID);
                HDR.H1 = [HDR.H1, line];
                if strncmp('"NCHAN%"',line,8) 
                        [tmp,line] = strtok(line,'=');
                        [tmp,line] = strtok(line,delim);
                        HDR.NS = str2double(char(tmp));
                end
                if strncmp('"NUM.POINTS"',line,12) 
                        [tmp,line] = strtok(line,'=');
                        [tmp,line] = strtok(line,delim);
                        HDR.SPR = str2double(tmp);
                end
                if strncmp('"ACT.FREQ"',line,10) 
                        [tmp,line] = strtok(line,'=');
                        [tmp,line] = strtok(line,delim);
                        HDR.SampleRate= str2double(tmp);
                end
                if strncmp('"DATE$"',line,7)
                        [tmp,line] = strtok(line,'=');
                        [date,line] = strtok(line,delim);
                        [tmp,date]=strtok(date,'-');
                        HDR.T0(3) = str2double(tmp);
                        [tmp,date]=strtok(date,'-');
                        HDR.T0(2) = str2double(tmp);
                        [tmp,date]=strtok(date,'-');
                        HDR.T0(1) = str2double(tmp);
                end
                if strncmp('"TIME$"',line,7)
                        [tmp,line] = strtok(line,'=');
                        [time,line] = strtok(line,delim);
                        [tmp,date]=strtok(time,':');
                        HDR.T0(4) = str2double(tmp);
                        [tmp,date]=strtok(date,':');
                        HDR.T0(5) = str2double(tmp);
                        [tmp,date]=strtok(date,':');
                        HDR.T0(6) = str2double(tmp);
                end;
                if strncmp('"UNITS$[]"',line,10)
                        [tmp,line] = strtok(char(line),'=');
                        for k=1:HDR.NS,
                                [tmp,line] = strtok(line,[' ,',delim]);
                                HDR.PhysDim(k,1:length(tmp)) = tmp;
                        end;
                end
                if strncmp('"CHANNEL.RANGES[]"',line,18)
                        [tmp,line] = strtok(line,'= ');
                        [tmp,line] = strtok(line,'= ');
                        for k=1:HDR.NS,
                                [tmp,line] = strtok(line,[' ',delim]);
                                [tmp1, tmp]=strtok(tmp,'(),');
                                HDR.PhysMin(k,1)=str2double(tmp1);
                                [tmp2, tmp]=strtok(tmp,'(),');
                                HDR.PhysMax(k,1)=str2double(tmp2);
                        end;
                end
                if strncmp('"CHAN$[]"',line,9)
                        [tmp,line] = strtok(line,'=');
                        for k=1:HDR.NS,	
                                [tmp,line] = strtok(line,[' ,',delim]);
                                HDR.Label(k,1:length(tmp)) = char(tmp);
                        end;
                end
                if 0,strncmp('"CHANNEL.LABEL$[]"',line,18)
                        [tmp,line] = strtok(line,'=');
                        for k=1:HDR.NS,
                                [HDR.Label{k,1},line] = strtok(line,delim);
                        end;
                end
                if strncmp(line,'"TR"',4) 
                        HDR.H1 = HDR.H1(1:length(HDR.H1)-length(line));
                        line = fgetl(HDR.FILE.FID); % get the time and date stamp line
                        tmp=fread(HDR.FILE.FID,1,'uint8'); % read sync byte hex-AA char
                        if tmp~=hex2dec('AA');
                                fprintf(HDR.FILE.stderr,'Error SOPEN type=SMA: Sync byte is not "AA"\n');
                        end;        
                        numbegin=1;
                end
        end
        
        %%%%%%%%%%%%%%%%%%% check file length %%%%%%%%%%%%%%%%%%%%
        
        HDR.FILE.POS= 0;
        HDR.HeadLen = ftell(HDR.FILE.FID);  % Length of Header
        fseek(HDR.FILE.FID,0,'eof'); 
        endpos = ftell(HDR.FILE.FID); 

        fclose(HDR.FILE.FID);
        PERMISSION = PERMISSION(PERMISSION~='t');       % open in binary mode 
        HDR.FILE.FID = fopen(HDR.FileName,[PERMISSION,'b'],'ieee-le');
        
        fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
        %[HDR.AS.endpos,HDR.HeadLen,HDR.NS,HDR.SPR,HDR.NS*HDR.SPR*4,HDR.AS.endpos-HDR.HeadLen - HDR.NS*HDR.SPR*4]
        HDR.AS.endpos = HDR.NS*HDR.SPR*4 - HDR.HeadLen;
        if endpos-HDR.HeadLen ~= HDR.NS*HDR.SPR*4;
                fprintf(HDR.FILE.stderr,'Warning SOPEN TYPE=SMA: Header information does not fit size of file\n');
                fprintf(HDR.FILE.stderr,'\tProbably more than one data segment - this is not supported in the current version of SOPEN\n');
        end
        HDR.AS.bpb    = HDR.NS*4;
        HDR.AS.endpos = (HDR.AS.endpos-HDR.HeadLen)/HDR.AS.bpb;
        HDR.Dur = 1/HDR.SampleRate;
        HDR.NRec = 1;
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);

        if ~isfield(HDR,'SMA')
                HDR.SMA.EVENT_CHANNEL= 1;
                HDR.SMA.EVENT_THRESH = 2.3;
        end;
        HDR.Filter.T0 = zeros(1,length(HDR.SMA.EVENT_CHANNEL));
        
        
elseif strcmp(HDR.TYPE,'RDF'),  % UCSD ERPSS acqusition software DIGITIZE
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
        
        status = fseek(HDR.FILE.FID,4,-1);
        HDR.FLAG.compressed = fread(HDR.FILE.FID,1,'uint16');
        HDR.NS = fread(HDR.FILE.FID,1,'uint16');
        status = fseek(HDR.FILE.FID,552,-1);
        HDR.SampleRate  = fread(HDR.FILE.FID,1,'uint16');
        status = fseek(HDR.FILE.FID,580,-1);
        tmp = fread(HDR.FILE.FID,[8,HDR.NS],'char');
        HDR.Label = char(tmp');
        
        cnt = 0;
        ev_cnt = 0;
        ev = [];
        
        % first pass, scan data
        totalsize = 0;
        tag = fread(HDR.FILE.FID,1,'uint32');
        while ~feof(HDR.FILE.FID) %& ~status,
                if tag == hex2dec('f0aa55'),
                        cnt = cnt + 1;
                        HDR.Block.Pos(cnt) = ftell(HDR.FILE.FID);
                        
                        % Read nchans and block length
                        tmp = fread(HDR.FILE.FID,34,'uint16');
                        
                        %fseek(HDR.FILE.FID,2,0);
                        nchans = tmp(2); %fread(HDR.FILE.FID,1,'uint16');
                        %fread(HDR.FILE.FID,1,'uint16');
                        block_size = 2^tmp(3); %fread(HDR.FILE.FID,1,'uint16');
                        blocksize2 = tmp(4);
                        %ndupsamp = fread(HDR.FILE.FID,1,'uint16');
                        %nrun = fread(HDR.FILE.FID,1,'uint16');
                        %err_detect = fread(HDR.FILE.FID,1,'uint16');
                        %nlost = fread(HDR.FILE.FID,1,'uint16');
                        HDR.EVENT.N = tmp(9); %fread(HDR.FILE.FID,1,'uint16');
                        %fseek(HDR.FILE.FID,50,0);

                        % Read events
                        HDR.EVENT.POS = repmat(nan,HDR.EVENT.N,1);
                        HDR.EVENT.TYP = repmat(nan,HDR.EVENT.N,1);
                        for i = 1:HDR.EVENT.N,
                                tmp = fread(HDR.FILE.FID,2,'uint8');
                                %cond_code = fread(HDR.FILE.FID,1,'uint8');
                                ev_code = fread(HDR.FILE.FID,1,'uint16');
                                ev_cnt  = ev_cnt + 1;
                                tmp2.sample_offset = tmp(1) + (cnt-1)*128;
                                tmp2.cond_code     = tmp(2);
                                tmp2.event_code    = ev_code;
                                if ~exist('OCTAVE_VERSION','builtin'), 	   
                                        ev{ev_cnt} = tmp2;
                                end;
                                HDR.EVENT.POS(ev_cnt) = tmp(1) + (cnt-1)*128;
                                HDR.EVENT.TYP(ev_cnt) = ev_code;
                        end;
                        status = fseek(HDR.FILE.FID,4*(110-HDR.EVENT.N)+2*nchans*block_size,0);
                else
                        [tmp, c] = fread(HDR.FILE.FID,3,'uint16');
			if (c > 2),
	                        nchans = tmp(2); %fread(HDR.FILE.FID,1,'uint16');
    		                block_size = 2^tmp(3); %fread(HDR.FILE.FID,1,'uint16');

            		        %fseek(HDR.FILE.FID,62+4*(110-HDR.EVENT.N)+2*nchans*block_size,0);
				sz = 62 + 4*110 + 2*nchans*block_size;
				status = -(sz>=(2^31));
				if ~status,
				        status = fseek(HDR.FILE.FID, sz, 0);
				end;	
			end;
                end
                tag = fread(HDR.FILE.FID,1,'uint32');
        end
        HDR.NRec = cnt;
        
        HDR.Events = ev;
        HDR.HeadLen = 0;
        HDR.FLAG.TRIGGERED = 1;	        
        HDR.FILE.POS = 0; 
        HDR.SPR = block_size;
        HDR.AS.bpb = HDR.SPR*HDR.NS*2;
        HDR.Dur = HDR.SPR/HDR.SampleRate;
	HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
        
        
elseif strcmp(HDR.TYPE,'LABVIEW'),
        HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-be');
        
        tmp = fread(HDR.FILE.FID,8,'uchar');
        HDR.VERSION = char(fread(HDR.FILE.FID,[1,8],'uchar'));
        HDR.AS.endpos = fread(HDR.FILE.FID,1,'int32'); % 4 first bytes = total header length
        
        HDR.HeadLen  = fread(HDR.FILE.FID,1,'int32'); % 4 first bytes = total header length
        HDR.NS       = fread(HDR.FILE.FID,1,'int32');  % 4 next bytes = channel list string length
        HDR.AS.endpos2 = fread(HDR.FILE.FID,1,'int32'); % 4 first bytes = total header length
        
        HDR.ChanList = fread(HDR.FILE.FID,HDR.NS,'uchar'); % channel string
        
        fclose(HDR.FILE.FID);
        %HDR.FILE.OPEN = 1;
        HDR.FILE.FID = -1;
        
        return;
        
        %%%%% READ HEADER from Labview 5.1 supplied VI "create binary header"
        
        HDR.HeadLen  = fread(HDR.FILE.FID,1,'int32'); % 4 first bytes = total header length
        HDR.NS     = fread(HDR.FILE.FID,1,'int32');  % 4 next bytes = channel list string length
        HDR.ChanList = fread(HDR.FILE.FID,HDR.NS,'uchar'); % channel string
        
        % Number of channels = 1 + ord(lastChann) - ord(firstChann):
        HDR.LenN     = fread(HDR.FILE.FID,1,'int32'); % Hardware config length
        HDR.HWconfig = fread(HDR.FILE.FID,HDR.LenN,'uchar'); % its value
        HDR.SampleRate = fread(HDR.FILE.FID,1,'float32');
        HDR.InterChannelDelay = fread(HDR.FILE.FID,1,'float32');
        tmp=fread(HDR.FILE.FID,[1,HDR.HeadLen - ftell(HDR.FILE.FID)],'uchar'); % read rest of header
        [HDR.Date,tmp]= strtok(tmp,9) ; % date is the first 10 elements of this tmp array (strip out tab)
        [HDR.Time,tmp]= strtok(tmp,9); % and time is the next 8 ones
        % HDR.T0 = [yyyy mm dd hh MM ss];   %should be Matlab date/time format like in clock()
        HDR.Description= char(tmp); % description is the rest of elements.
        
        % Empirically determine the number of bytes per multichannel point:
        HDR.HeadLen = ftell(HDR.FILE.FID) ; 
        dummy10 = fread(HDR.FILE.FID,[HDR.NS,1],'int32');
        HDR.AS.bpb = (ftell(HDR.FILE.FID) - HDR.HeadLen); % hope it's an int !
        
        tmp = fseek(HDR.FILE.FID,0,'eof'); 
        HDR.AS.endpos = (ftell(HDR.FILE.FID) - HDR.HeadLen)/HDR.AS.bpb;
        fseek(HDR.FILE.FID,HDR.HeadLen,'bof'); 
        
        HDR.Cal = 1;
        
        
elseif strcmp(HDR.TYPE,'RG64'),
        fid = fopen(HDR.FileName,PERMISSION,'ieee-le');
        
        HDR.IDCODE=char(fread(fid,[1,4],'char'));	%
        if ~strcmp(HDR.IDCODE,'RG64') 
                fprintf(HDR.FILE.stderr,'\nError LOADRG64: %s not a valid RG64 - header file\n',HDR.FileName); 
                HDR.TYPE = 'unknown';
                fclose(fid);
                return;
        end; %end;
        
        tmp = fread(fid,2,'int32');
        HDR.VERSION = tmp(1)+tmp(2)/100;
        HDR.NS = fread(fid,1,'int32');
        HDR.SampleRate = fread(fid,1,'int32');
        HDR.SPR = fread(fid,1,'int32')/HDR.NS;
        AMPF = fread(fid,64,'int32');		
        fclose(fid);
        
        HDR.HeadLen = 0;
        HDR.PhysDim = 'uV';
        HDR.Cal = (5E6/2048)./AMPF;
        HDR.AS.endpos = HDR.SPR;
        HDR.AS.bpb    = HDR.NS*2;
        HDR.GDFTYP    = 'int16';
        
        EXT = HDR.FILE.Ext; 
        if upper(EXT(2))~='D',
                EXT(2) = EXT(2) - 'H' + 'D';
        end;
        FILENAME=fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.',EXT]);
        
        HDR.FILE.FID=fopen(FILENAME,'rb','ieee-le');
        if HDR.FILE.FID<0,
                fprintf(HDR.FILE.stderr,'\nError LOADRG64: data file %s not found\n',FILENAME); 
                return;
        end;

        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal(1:HDR.NS),HDR.NS+1,HDR.NS);
        HDR.FILE.POS = 0; 
        HDR.FILE.OPEN= 1;
        
        
elseif strcmp(HDR.TYPE,'DDF'),
        
        % implementation of this format is not finished yet.
        fprintf(HDR.FILE.stderr,'Warning SOPEN: Implementing DASYLAB format not completed yet. Contact <a.schloegl@ieee.org> if you are interested in this feature.\n');
        %HDR.FILE.FID = -1;
        %return;
        
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS = 0;
                %HDR.ID = fread(HDR.FILE.FID,5,'char');
                ds=fread(HDR.FILE.FID,[1,128],'char');
                HDR.ID = setstr(ds(1:5));
                DataSource = ds;
                k = 0;
                while ~(any(ds==26)),
                        ds = fread(HDR.FILE.FID,[1,128],'char');
                        DataSource = [DataSource,ds];
                        k = k+1;	
                end;	
                pos = find(ds==26)+k*128;
                DataSource = setstr(DataSource(6:pos));
                HDR.DDF.Source = DataSource;
                while ~isempty(DataSource),
                        [ds,DataSource] = strtok(setstr(DataSource),[10,13]);
                        [field,value] = strtok(ds,'=');
                        if strfind(field,'SAMPLE RATE');
                                [tmp1,tmp2] = strtok(value,'=');
                                HDR.SampleRate = str2double(tmp1);
                        elseif strfind(field,'DATA CHANNELS');
                                HDR.NS = str2double(value);
                        elseif strfind(field,'START TIME');
                                Time = value;
                        elseif strfind(field,'DATA FILE');
                                HDR.FILE.DATA = value;
                        end;			 	
                end;
                fseek(HDR.FILE.FID,pos,'bof'); 	% position file identifier
                if 0;%DataSource(length(DataSource))~=26,
                        fprintf(1,'Warning: DDF header seems to be incorrenct. Contact <alois.schloegl@tugraz.at> Subject: BIOSIG/DATAFORMAT/DDF  \n');
                end;
                HDR.DDF.CPUidentifier  = setstr(fread(HDR.FILE.FID,[1,2],'char'));
                HDR.HeadLen(1) = fread(HDR.FILE.FID,1,'uint16');
                tmp = fread(HDR.FILE.FID,1,'uint16');
                if tmp == 0, HDR.GDFTYP = 'uint16'; 		% streamer format (data are raw data in WORD=UINT16)
                elseif tmp == 1, HDR.GDFTYP = 'float32'; 	% Universal Format 1 (FLOAT)
                elseif tmp == 2, HDR.GDFTYP = 'float64'; 	% Universal Format 2 (DOUBLE)
                elseif tmp <= 1000, % reserved
                else		% unused
                end;
                HDR.FILE.Type  = tmp;
                HDR.VERSION    = fread(HDR.FILE.FID,1,'uint16');
                HDR.HeadLen(2) = fread(HDR.FILE.FID,1,'uint16');	% second global Header
                HDR.HeadLen(3) = fread(HDR.FILE.FID,1,'uint16');	% size of channel Header
                fread(HDR.FILE.FID,1,'uint16');	% size of a block Header
                tmp = fread(HDR.FILE.FID,1,'uint16');
                if tmp ~= isfield(HDR.FILE,'DATA')
                        fprintf(1,'Warning: DDF header seems to be incorrenct. Contact <alois.schloegl@tugraz.at> Subject: BIOSIG/DATAFORMAT/DDF  \n');
                end;
                HDR.NS = fread(HDR.FILE.FID,1,'uint16');
                HDR.Delay = fread(HDR.FILE.FID,1,'double');
                HDR.StartTime = fread(HDR.FILE.FID,1,'uint32');  % might be incorrect
                
                % it looks good so far. 
                % fseek(HDR.FILE.FID,HDR.HeadLen(1),'bof');
                if HDR.FILE.Type==0,
                        % second global header
                        fread(HDR.FILE.FID,1,'uint16')	% overall number of bytes in this header
                        fread(HDR.FILE.FID,1,'uint16')	% number of analog channels
                        fread(HDR.FILE.FID,1,'uint16')	% number of counter channels
                        fread(HDR.FILE.FID,1,'uint16')	% number of digital ports
                        fread(HDR.FILE.FID,1,'uint16')	% number of bits in each digital port
                        fread(HDR.FILE.FID,1,'uint16')	% original blocksize when data was stored
                        fread(HDR.FILE.FID,1,'uint32')	% sample number of the first sample (when cyclic buffer not activated, always zero
                        fread(HDR.FILE.FID,1,'uint32')	% number of samples per channel
                        
                        % channel header
                        for k = 1:HDR.NS,
                                fread(HDR.FILE.FID,1,'uint16')	% number of bytes in this hedader
                                fread(HDR.FILE.FID,1,'uint16')	% channel type 0: analog, 1: digital, 2: counter
                                HDR.Label = setstr(fread(HDR.FILE.FID,[24,16],'char')');	% 
                                tmp = fread(HDR.FILE.FID,1,'uint16')	% dataformat 0 UINT, 1: INT
                                HDR.GDFTYP(k) = 3 + (~tmp);
                                HDR.Cal(k) = fread(HDR.FILE.FID,1,'double');	% 
                                HDR.Off(k) = fread(HDR.FILE.FID,1,'double');	% 
                        end;    
                        
                elseif HDR.FILE.Type==1,
                        % second global header
                        HDR.pos1 = ftell(HDR.FILE.FID);
                        tmp = fread(HDR.FILE.FID,1,'uint16');	% size of this header 
                        if (tmp~=HDR.HeadLen(2)),
                                fprintf(HDR.FILE.stderr,'Error SOPEN DDF: error in header of file %s\n',HDR.FileName);
                        end;
                        HDR.U1G.NS = fread(HDR.FILE.FID,1,'uint16');	% number of channels
                        HDR.FLAG.multiplexed = fread(HDR.FILE.FID,1,'uint16');	% multiplexed: 0=no, 1=yes
                        HDR.DDF.array = fread(HDR.FILE.FID,[1,16],'uint16');	% array of channels collected on each input channel
                        
                        % channel header
                        for k = 1:HDR.NS,
                                filepos = ftell(HDR.FILE.FID);    
                                taglen = fread(HDR.FILE.FID,1,'uint16');	% size of this header
                                ch = fread(HDR.FILE.FID,1,'uint16');	% channel number
                                HDR.DDF.MAXSPR(ch+1)= fread(HDR.FILE.FID,1,'uint16');	% maximum size of block in samples
                                HDR.DDF.delay(ch+1) = fread(HDR.FILE.FID,1,'double');	% time delay between two samples
                                HDR.DDF.ChanType(ch+1) = fread(HDR.FILE.FID,1,'uint16');	% channel type 
                                HDR.DDF.ChanFlag(ch+1) = fread(HDR.FILE.FID,1,'uint16');	% channel flag 
                                unused = fread(HDR.FILE.FID,2,'double');	% must be 0.0 for future extension
                                tmp = fgets(HDR.FILE.FID);	% channel unit
                                HDR.PhysDim{k} = [tmp,' '];	% channel unit
                                tmp = fgets(HDR.FILE.FID);		% channel name 
                                HDR.Label{k} = [tmp,' '];		% channel name 
                                fseek(HDR.FILE.FID,filepos+taglen,'bof');
                        end;
                        HDR.PhysDim = strvcat(HDR.PhysDim);	% channel unit
                        HDR.Label   = strvcat(HDR.Label);		% channel name 
                        
                        % channel header
                        for k = 1:HDR.NS,
                                fread(HDR.FILE.FID,[1,4],'char');
                                fread(HDR.FILE.FID,1,'uint16');	% overall number of bytes in this header
                                HDR.BlockStartTime = fread(HDR.FILE.FID,1,'uint32');  % might be incorrect
                                unused = fread(HDR.FILE.FID,2,'double');	% must be 0.0 for future extension
                                ch = fread(HDR.FILE.FID,1,'uint32');  % channel number
                        end;    
                        fseek(HDR.FILE.FID,HDR.pos1+sum(HDR.HeadLen(2:3)),'bof');
                        
                elseif HDR.FILE.Type==2,
                        % second global header
                        pos = ftell(HDR.FILE.FID);
                        HeadLen = fread(HDR.FILE.FID,1,'uint16');	% size of this header 
                        fread(HDR.FILE.FID,1,'uint16');	% number of channels
                        fseek(HDR.FILE.FID, pos+HeadLen ,'bof');
                        
                        % channel header
                        for k = 1:HDR.NS,
                                pos = ftell(HDR.FILE.FID);
                                HeadLen = fread(HDR.FILE.FID,1,'uint16');	% size of this header 
                                HDR.DDF.Blocksize(k) = fread(HDR.FILE.FID,1,'uint16');	% 
                                HDR.DDF.Delay(k) = fread(HDR.FILE.FID,1,'double');	% 
                                HDR.DDF.chantyp(k) = fread(HDR.FILE.FID,1,'uint16');	% 
                                HDR.FLAG.TRIGGER(k) = ~~fread(HDR.FILE.FID,1,'uint16');	
                                fread(HDR.FILE.FID,1,'uint16');	
                                HDR.Cal(k) = fread(HDR.FILE.FID,1,'double');	
                        end;
                else
                        
                end;
                %ftell(HDR.FILE.FID),
                tag=fread(HDR.FILE.FID,[1,4],'char');
        end;
        return;         

        
elseif strcmp(HDR.TYPE,'MIT')
        if any(PERMISSION=='r'),
                HDR.FileName = fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.',HDR.FILE.Ext]);
                
                HDR.FILE.FID = fopen(HDR.FileName,'r','ieee-le');
                if HDR.FILE.FID<0,
			fprintf(HDR.FILE.stderr,'Error SOPEN: Couldnot open file %s\n',HDR.FileName);
			return;
		end;	
                
                fid = HDR.FILE.FID;
                z   = fgetl(fid);
                while strncmp(z,'#',1) | isempty(z),
                        z   = fgetl(fid);
                end;
                tmpfile = strtok(z,' /');
                if ~strcmpi(HDR.FILE.Name,tmpfile),
                        fprintf(HDR.FILE.stderr,'Error: RecordName %s does not fit filename %s\n',tmpfile,HDR.FILE.Name);
                        fclose(HDR.FILE.FID)
                        return; 
                end;	
                
                %A = sscanf(z, '%*s %d %d %d',[1,3]);
		t = z;
		k = 0;
		while ~isempty(t)
			k = k + 1;
	                [s,t] = strtok(t,[9,10,13,32]);
			Z{k}  = s;
			if any(s==':'),
                                t0 = str2double(s,':');
				HDR.T0(3+(1:length(t0))) = t0;
			elseif sum(s=='/')==2,
				HDR.T0([3,2,1])=str2double(s,'/');
			end;	
		end;	
                HDR.NS   = str2double(Z{2});   % number of signals

		if k>2,
	                [tmp,tmp1] = strtok(Z{3},'/');
	                HDR.SampleRate = str2double(tmp);   % sample rate of data
		end;

		[tmp,z1] = strtok(Z{1},'/');
		if ~isempty(z1)	
			%%%%%%%%%% Multi-Segment files %%%%%%% 
			fprintf(HDR.FILE.stderr,'Error SOPEN (MIT) %s:  multi-segment files not supported.\n',tmpfile);
			
			return;
			
			HDR.FLAG.TRIGGERED = 1; 
			z1 = strtok(z1,'/');
			HDR.NRec = str2double(z1);
			
			HDR.EVENT.TYP = repmat(hex2dec('0300'),HDR.NRec,1);
			HDR.EVENT.POS = repmat(NaN,HDR.NRec,1);
			HDR.EVENT.DUR = repmat(NaN,HDR.NRec,1);
			HDR.EVENT.CHN = repmat(0,HDR.NRec,1);
			count = 0; 
			for k = 1:HDR.NRec;
				[s,t] = strtok(fgetl(fid));
				[hdr] = sopen(fullfile(HDR.FILE.Path,[s,'.hea']),'r',CHAN);
				[s,hdr] = sread(hdr);
				hdr = sclose(hdr);
				if k==1, 
					HDR.data = repmat(s,HDR.NRec,1); 
				else 
					HDR.data(count+1:count+size(s,1),:) = s;
				end;
				HDR.EVENT.POS(k) = count;
				HDR.EVENT.DUR(k) = size(s,1);
				count = count + size(s,1);
			end;
			HDR.Label = hdr.Label;
			HDR.PhysDim = hdr.PhysDim; 
			HDR.SPR = size(s,1);
			HDR.NS  = hdr.NS;
			HDR.Calib = (hdr.Calib>0);
			HDR.FLAG.TRIGGERED = 1; 
			HDR.FILE.POS = 0;
			HDR.TYPE = 'native';

		else

    		        [tmp,z] = strtok(z); 
    		        [tmp,z] = strtok(z);
	                %HDR.NS  = str2double(Z{2});   % number of signals
	                [tmp,z] = strtok(z);
	                [tmp,z] = strtok(z,' ()');
	                HDR.NRec = str2double(tmp);   % length of data
                        HDR.SPR = 1; 
                
	                HDR.MIT.gain = zeros(1,HDR.NS);
			HDR.MIT.zerovalue  = repmat(NaN,1,HDR.NS);
			HDR.MIT.firstvalue = repmat(NaN,1,HDR.NS);
	                for k = 1:HDR.NS,
	                        z = fgetl(fid);
	                        [HDR.FILE.DAT{k,1},z]=strtok(z);
	                        for k0 = 1:7,
	                                [tmp,z] = strtok(z);
	                                if k0 == 1, 
	                                        [tmp, tmp1] = strtok(tmp,'x:');
	                                        [tmp, status] = str2double(tmp); 
	                                        HDR.MIT.dformat(k,1) = tmp;
                                                HDR.AS.SPR(k) = 1; 
	                                        if isempty(tmp1)
                                                elseif tmp1(1)=='x'
	                                                HDR.AS.SPR(k) = str2double(tmp1(2:end)); 
	                                        elseif tmp1(1)==':'
                                                        fprintf(HDR.FILE.stderr,'Warning SOPEN: skew information in %s is ignored.\n', HDR.FileName);
	                                        end                                                
	                                elseif k0==2,  
	                                        % EC13*.HEA files have special gain values like "200(23456)/uV". 
	                                        [tmp, tmp2] = strtok(tmp,'/');
						tmp2 = [tmp2(2:end),' '];
	                                        HDR.PhysDim(k,1:length(tmp2)) = tmp2;
	                                        [tmp, tmp1] = strtok(tmp,' ()');
	                                        [tmp, status] = str2double(tmp); 
	                                        if isempty(tmp), tmp = 0; end;   % gain
	                                        if isnan(tmp),   tmp = 0; end;
	                                        HDR.MIT.gain(1,k) = tmp;
	                                elseif k0==3,
	                                        [tmp, status] = str2double(tmp); 
	                                        if isempty(tmp), tmp = NaN; end; 
	                                        if isnan(tmp),   tmp = NaN; end;
	                                        HDR.Bits(1,k) = tmp;
	                                elseif k0==4,
	                                        [tmp, status] = str2double(tmp);
	                                        if isempty(tmp), tmp = 0; end;
	                                        if isnan(tmp),   tmp = 0; end;
	                                        HDR.MIT.zerovalue(1,k) = tmp; 
	                                elseif k0==5, 
	                                        [tmp, status] = str2double(tmp);
	                                        if isempty(tmp), tmp = NaN; end; 
	                                        if isnan(tmp),   tmp = NaN; end;
	                                        HDR.MIT.firstvalue(1,k) = tmp;        % first integer value of signal (to test for errors)
	                                else
	                                        
	                                end;
	                        end;
	                        HDR.Label(k,1:length(z)+1) = [z,' ']; 
	                end;
	                
	                HDR.MIT.gain(HDR.MIT.gain==0) = 200;    % default gain 
	                HDR.Calib = sparse([HDR.MIT.zerovalue; eye(HDR.NS)]*diag(1./HDR.MIT.gain(:)));
	                HDR.Label = char(HDR.Label);
	                
	                z = char(fread(fid,[1,inf],'char'));
	                ix1 = [strfind(upper(z),'AGE:')+4, strfind(upper(z),'AGE>:')+5];
	                if ~isempty(ix1),
	                        [tmp,z]=strtok(z(ix1(1):length(z)));
	                        HDR.Patient.Age = str2double(tmp);
	                end;
	                ix1 = [strfind(upper(z),'SEX:')+4, strfind(upper(z),'SEX>:')+5];
	                if ~isempty(ix1),
	                        [HDR.Patient.Sex,z]=strtok(z(ix1(1):length(z)));
	                end;
	                ix1 = [strfind(upper(z),'BMI:')+4, strfind(upper(z),'BMI>:')+5];
	                if ~isempty(ix1),
	                        [tmp,z]=strtok(z(ix1(1):length(z)));
	                        HDR.Patient.BMI = str2double(tmp);
	                end;
	                ix1 = [strfind(upper(z),'DIAGNOSIS:')+10; strfind(upper(z),'DIAGNOSIS>:')+11];
	                if ~isempty(ix1),
	                        [HDR.Patient.Diagnosis,z]=strtok(z(ix1(1):length(z)),char([10,13,abs('#<>')]));
	                end;
	                ix1 = [strfind(upper(z),'MEDICATIONS:')+12, strfind(upper(z),'MEDICATIONS>:')+13];
	                if ~isempty(ix1),
	                        [HDR.Patient.Medication,z]=strtok(z(ix1(1):length(z)),char([10,13,abs('#<>')]));
	                end;
	                fclose(fid);
	
			%------ LOAD ATR FILE ---------------------------------------------------                        
			tmp = fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.atr']);
			if ~exist(tmp,'file'),
				tmp = fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.ATR']);
			end;
			if exist(tmp,'file'),
		                H = sopen(tmp);
				HDR.EVENT = H.EVENT; 
				HDR.EVENT.SampleRate = HDR.SampleRate;
			end;
	
	                %------ LOAD BINARY DATA --------------------------------------------------
	                if ~HDR.NS, 
	                        return; 
	                end;
	                if all(HDR.MIT.dformat==HDR.MIT.dformat(1)),
	                        HDR.VERSION = HDR.MIT.dformat(1);
	                else
	                        fprintf(HDR.FILE.stderr,'Error SOPEN: different DFORMATs not supported.\n');
	                        HDR.FILE.FID = -1;
	                        return;
	                end;

			GDFTYP = repmat(NaN,HDR.NS,1);
			GDFTYP(HDR.MIT.dformat==80) = 2;
			GDFTYP(HDR.MIT.dformat==16) = 3;
			GDFTYP(HDR.MIT.dformat==61) = 3;
			GDFTYP(HDR.MIT.dformat==160)= 4;
			GDFTYP(HDR.MIT.dformat==212)= 255+12;
			GDFTYP(HDR.MIT.dformat==310)= 255+10;
			GDFTYP(HDR.MIT.dformat==311)= 255+10;
			if ~any(isnan(GDFTYP)), HDR.GDFTYP = GDFTYP; end; 
			HDR.RID = HDR.FILE(1).Name;
			HDR.PID = ''; 

	                HDR.AS.spb = sum(HDR.AS.SPR);
	                if 0,
	                        
	                elseif HDR.VERSION == 212, 
	                        HDR.AS.bpb = HDR.AS.spb*3/2;
	                elseif HDR.VERSION == 310, 
	                        HDR.AS.bpb = HDR.AS.spb/3*4;
	                elseif HDR.VERSION == 311, 
	                        HDR.AS.bpb = HDR.AS.spb/3*4;
	                elseif HDR.VERSION == 8, 
	                        HDR.AS.bpb = HDR.AS.spb;
	                elseif HDR.VERSION == 80, 
	                        HDR.AS.bpb = HDR.AS.spb;
	                elseif HDR.VERSION == 160, 
	                        HDR.AS.bpb = HDR.AS.spb;
    	                elseif HDR.VERSION == 16, 
	                        HDR.AS.bpb = HDR.AS.spb;
	                elseif HDR.VERSION == 61, 
	                        HDR.AS.bpb = HDR.AS.spb;
	                end;
			if HDR.AS.bpb==round(HDR.AS.bpb),
				d = 1; 
			else
				[HDR.AS.bpb,d] = rat(HDR.AS.bpb);
				HDR.NRec   = HDR.NRec/d; 
				HDR.AS.SPR = HDR.AS.SPR*d;
				HDR.AS.spb = HDR.AS.spb*d;
			end;
	                HDR.AS.bi = [0;cumsum(HDR.AS.SPR(:))]; 
	                HDR.SPR = HDR.AS.SPR(1);
	                for k = 2:HDR.NS,
	                        HDR.SPR = lcm(HDR.SPR,HDR.AS.SPR(k));
	                end;
	                HDR.AS.SampleRate = HDR.SampleRate*HDR.AS.SPR/d;
	                HDR.SampleRate = HDR.SampleRate*HDR.SPR/d;
	                
			
	                HDR.Dur = HDR.SPR/HDR.SampleRate;
	
	                if HDR.VERSION ==61,
	                        MACHINE_FORMAT='ieee-be';
	                else
	                        MACHINE_FORMAT='ieee-le';
	                end;
	
			DAT = strvcat(HDR.FILE.DAT);
			if all(all(DAT == DAT(ones(size(DAT,1),1),:))),
				% single DAT-file: only this provides high performance 
				HDR.FILE.DAT = DAT(1,:);
	
	            		tmpfile = fullfile(HDR.FILE.Path,HDR.FILE.DAT);
	                	if  ~exist(tmpfile,'file'), 
	                	        HDR.FILE.DAT = upper(HDR.FILE.DAT);
	                	        tmpfile = fullfile(HDR.FILE.Path,HDR.FILE.DAT);
	                	end;
	                	if  ~exist(tmpfile,'file'), 
	                	        HDR.FILE.DAT = lower(HDR.FILE.DAT);
	                	        tmpfile = fullfile(HDR.FILE.Path,HDR.FILE.DAT);
	                	end;
	                	HDR.FILE.FID = fopen(tmpfile,'rb',MACHINE_FORMAT);
	                	if HDR.FILE.FID<0,
					fprintf(HDR.FILE.stderr,'Error SOPEN: Couldnot open file %s\n',tmpfile);
					return;
				end;	
		
		                HDR.FILE.OPEN = 1;
		                HDR.FILE.POS  = 0;
		                HDR.HeadLen   = 0;
		                status = fseek(HDR.FILE.FID,0,'eof');
		                tmp = ftell(HDR.FILE.FID);
		                try
		                        HDR.AS.endpos = tmp/HDR.AS.bpb;
		                catch
		                        fprintf(HDR.FILE.stderr,'Warning 2003 SOPEN: FTELL does not return numeric value (Octave > 2.1.52).\nHDR.AS.endpos not completed.\n');
		                end;
		                status = fseek(HDR.FILE.FID,0,'bof');
	
		                HDR.InChanSelect = 1:HDR.NS;
		                FLAG_UCAL = HDR.FLAG.UCAL;	
		                HDR.FLAG.UCAL = 1;
		                S = NaN;
		                [S,HDR] = sread(HDR,HDR.SPR/HDR.SampleRate); % load 1st sample
		                if (HDR.VERSION>0) & (any(S(1,:) - HDR.MIT.firstvalue)), 
		                        fprintf(HDR.FILE.stderr,'Warning SOPEN MIT-ECG: First values of header and datablock do not fit in file %s.\n\tHeader:\t',HDR.FileName); 
		                        fprintf(HDR.FILE.stderr,'\t%5i',HDR.MIT.firstvalue);
		                        fprintf(HDR.FILE.stderr,'\n\tData 1:\t');
		                        fprintf(HDR.FILE.stderr,'\t%5i',S(1,:));
		                        fprintf(HDR.FILE.stderr,'\n');
		                end;
		                HDR.FLAG.UCAL = FLAG_UCAL ;	
		                fseek(HDR.FILE.FID,0,'bof');	% reset file pointer
	
	                else
				% Multi-DAT files 
				[i,j,k]=unique(HDR.FILE.DAT);
				for k1 = 1:length(j),
					ix = (k==k1);
					f = fullfile(HDR.FILE.Path,HDR.FILE.DAT{j(k1)});
					hdr.FILE.FID = fopen(f,'rb');
                                        if hdr.FILE.FID>0,
                                                hdr.FILE.stderr = HDR.FILE.stderr;
                                                hdr.FILE.stdout = HDR.FILE.stdout;
                                                hdr.FILE.POS = 0; 
                                                hdr.NS = sum(ix);
                                                hdr.InChanSelect = 1:hdr.NS;
                                                hdr.MIT.dformat = HDR.MIT.dformat(ix);
                                                %hdr.Calib = HDR.Calib(:,ix);
                                                hdr.AS.spb = sum(HDR.AS.SPR(ix));
                                                hdr.SampleRate = HDR.SampleRate;
                                                hdr.TYPE = 'MIT';
                                                hdr.SPR = HDR.SPR;
                                                hdr.AS.SPR = HDR.AS.SPR(ix);
                                                hdr.FLAG = HDR.FLAG; 
                                                hdr.FLAG.UCAL = 1; 
                                                
                                                if all(hdr.MIT.dformat(1)==hdr.MIT.dformat),
                                                        hdr.VERSION = hdr.MIT.dformat(1);
                                                else
                                                        fprintf(hdr.FILE.stderr,'different DFORMATs not supported.\n');
                                                        hdr.FILE.FID = -1;
                                                        return;
                                                end;
                                                if 0,
                                                        
                                                elseif hdr.VERSION == 212, 
                                                        if mod(hdr.AS.spb,2) 
                                                                hdr.AS.spb = hdr.AS.spb*2;
                                                        end
                                                        hdr.AS.bpb = hdr.AS.spb*3/2;
                                                elseif hdr.VERSION == 310, 
                                                        if mod(hdr.AS.spb,3) 
                                                                hdr.AS.spb = hdr.AS.spb*2/3;
                                                        end
                                                        hdr.AS.bpb = hdr.AS.spb*2;
                                                elseif hdr.VERSION == 311, 
                                                        if mod(hdr.AS.spb,3) 
                                                                hdr.AS.spb = hdr.AS.spb*3;
                                                        end
                                                        hdr.AS.bpb = hdr.AS.spb*4;
                                                elseif hdr.VERSION == 8, 
                                                        hdr.AS.bpb = hdr.AS.spb;
                                                elseif hdr.VERSION == 80, 
                                                        hdr.AS.bpb = hdr.AS.spb;
                                                elseif hdr.VERSION == 160, 
                                                        hdr.AS.bpb = hdr.AS.spb;
                                                elseif hdr.VERSION == 16, 
                                                        hdr.AS.bpb = hdr.AS.spb;
                                                elseif hdr.VERSION == 61, 
                                                        hdr.AS.bpb = hdr.AS.spb;
                                                end;
                                                [s,hdr] = sread(hdr);
                                                fclose(hdr.FILE.FID);
                                        else 
                                                s = [];
                                        end;
                                        
					if k1==1,
						HDR.data = s; 
					else	
						n = [size(s,1),size(HDR.data,1)];
						if any(n~=n(1)),
			    				fprintf(HDR.FILE.stderr,'Warning SOPEN MIT-ECG(%s): lengths of %s (%i) and %s (%i) differ\n',HDR.FileName,HDR.FILE.DAT{j(k1-1)},n(1),HDR.FILE.DAT{j(k1)},n(2));
						end;
						n = min(n);
						HDR.data = [HDR.data(1:n,:),s(1:n,:)];
					end;
				end;
				HDR.FILE.POS = 0; 
				HDR.TYPE = 'native';
			end; 
		end;
        end;
	        
	
elseif strcmp(HDR.TYPE,'MIT-ATR'),
                tmp = dir(fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.hea']));
		if isempty(tmp)
                        fprintf(HDR.FILE.stderr,'Warning SOPEN: no corresponing header file found for MIT-ATR EVENT file %s.\n',HDR.FileName);
		end;	
		
                %------ LOAD ATTRIBUTES DATA ----------------------------------------------
                fid = fopen(HDR.FileName,'rb','ieee-le');
                if fid<0,
                        A = 0; c = 0;
                else
                        [A,c] = fread(fid, inf, 'uint16');
                        fclose(fid);
                end;
		
		EVENTTABLE = repmat(NaN,c,3);
		Desc = repmat({''},ceil(c),1);
                FLAG63 = 0;
                K  = 0;
                i  = 1;
		ch = 0; 
		accu = 0; 

		tmp = floor(A(:)/1024);
		annoth = tmp;
		L   = A(:) - tmp*1024;
		tmp = floor(A(:)/256);
		t0  = char([A(:)-256*tmp, tmp])';
                while ((i<=size(A,1)) & (A(i)>0)),
			a = annoth(i);
                        if a==0,  % end of file
			  
                    	elseif a<50,
                            	K = K + 1;
	    			accu = accu + L(i);
				EVENTTABLE(K,:) = [a,accu,ch];
                        elseif a==59,	% SKIP 
				if (L(i)==0), 
					accu = accu + (2.^[0,16])*[A(i+2);A(i+1)];
    	        	                i = i + 2;
				else
					accu = accu + L(i);	
				end;	
			%elseif a==60,	% NUM
				%[60,L,A(i)]
                                % nothing to do!
                        %elseif a==61,	% SUB
				%[61,L,A(i)]
			        % nothing to do!
                        elseif a==62,	% CHN
				ch = L(i); 
                        elseif a==63,	% AUX
				c = ceil(L(i)/2);
				t = t0(:,i+1:i+c)';
				Desc{K} = t(:)'; 
                                FLAG63 = 1;
                        	i = i + c;
                        end;
                        i = i + 1;
                end;
		HDR.EVENT.TYP = EVENTTABLE(1:K,1); % + hex2dec('0540'); 
		HDR.EVENT.POS = EVENTTABLE(1:K,2); 
		HDR.EVENT.CHN = EVENTTABLE(1:K,3); 
		HDR.EVENT.DUR = zeros(K,1); 
                if FLAG63, HDR.EVENT.Desc = Desc(1:K); end;
		HDR.TYPE = 'EVENT';
                
        
elseif strcmp(HDR.TYPE,'TMS32'),        % Portilab/TMS32 format
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
                HDR.ID = fread(HDR.FILE.FID,31,'char');
                HDR.VERSION = fread(HDR.FILE.FID,1,'int16');
                [tmp,c] = fread(HDR.FILE.FID,81,'char');
                HDR.SampleRate = fread(HDR.FILE.FID,1,'int16');
                HDR.TMS32.StorageRate = fread(HDR.FILE.FID,1,'int16');
                HDR.TMS32.StorageType = fread(HDR.FILE.FID,1,'char');
                HDR.NS = fread(HDR.FILE.FID,1,'int16');
                HDR.AS.endpos = fread(HDR.FILE.FID,1,'int32');
                tmp = fread(HDR.FILE.FID,1,'int32');
                tmp = fread(HDR.FILE.FID,[1,7],'int16');
                HDR.T0   = tmp([1:3,5:7]);
                HDR.NRec = fread(HDR.FILE.FID,1,'int32');
                HDR.SPR  = fread(HDR.FILE.FID,1,'uint16');
                HDR.AS.bpb = fread(HDR.FILE.FID,1,'uint16')+86;
                HDR.FLAG.DeltaCompression = fread(HDR.FILE.FID,1,'int16');
                tmp = fread(HDR.FILE.FID,64,'char');
                HDR.HeadLen = 217 + HDR.NS*136;
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS = 0;
                
                Label   = zeros(HDR.NS,40);
                PhysDim = zeros(HDR.NS,10);
                k = 1;  aux = 0; 
                while k <= HDR.NS,
                        c   = fread(HDR.FILE.FID,[1,1],'uint8');
                        tmp = char(fread(HDR.FILE.FID,[1,40],'char'));
                        if strncmp(tmp,'(Lo)',4);
                                Label(k-aux,1:c-4) = tmp(5:c);
                                HDR.GDFTYP(k-aux)  = 16;
                        elseif strncmp(tmp,'(Hi)',4) ;
                                aux = aux + 1;				
                        else
                                Label(k-aux,1:c)  = tmp(1:c);
                                HDR.GDFTYP(k-aux) = 5;
                        end;
                        
                        tmp = fread(HDR.FILE.FID,[1,4],'char');
                        c   = fread(HDR.FILE.FID,[1,1],'uint8');
                        tmp = fread(HDR.FILE.FID,[1,10],'char');
                        PhysDim(k-aux,1:c) = tmp(1:c);
                        
                        HDR.PhysMin(k-aux,1) = fread(HDR.FILE.FID,1,'float32');			
                        HDR.PhysMax(k-aux,1) = fread(HDR.FILE.FID,1,'float32');			
                        HDR.DigMin(k-aux,1)  = fread(HDR.FILE.FID,1,'float32');			
                        HDR.DigMax(k-aux,1)  = fread(HDR.FILE.FID,1,'float32');			
                        HDR.TMS32.SI(k) = fread(HDR.FILE.FID,1,'int16');
                        tmp = fread(HDR.FILE.FID,2,'char');
                        tmp = fread(HDR.FILE.FID,60,'char');
                        k = k + 1;
                end;
                HDR.NS = HDR.NS-aux; 
                HDR.Label = deblank(char(Label(1:HDR.NS,:)));
                HDR.PhysDim = char(PhysDim(1:HDR.NS,:));
                HDR.Cal = (HDR.PhysMax-HDR.PhysMin)./(HDR.DigMax-HDR.DigMin);
                HDR.Off = HDR.PhysMin - HDR.Cal .* HDR.DigMin;
                HDR.Calib = sparse([HDR.Off';(diag(HDR.Cal))]);
        end;
        
        
elseif 0, strcmp(HDR.TYPE,'DAQ'),
        HDR = daqopen(HDR,PERMISSION);
        
        
elseif strcmp(HDR.TYPE,'MAT4') & any(PERMISSION=='r'),
    		HDR.FILE.FID = fopen(HDR.FileName,'rb',HDR.MAT4.opentyp);
                k=0; NB=0;
                %type = fread(HDR.FILE.FID,4,'uchar'); 	% 4-byte header
                type = fread(HDR.FILE.FID,1,'uint32'); 	% 4-byte header
                while ~isempty(type),
                        type = sprintf('%04i',type)';
                        type = type - abs('0');
                        k = k + 1;
                        [mrows,c] = fread(HDR.FILE.FID,1,'uint32'); 	% tag, datatype
                        ncols = fread(HDR.FILE.FID,1,'uint32'); 	% tag, datatype
                        imagf = fread(HDR.FILE.FID,1,'uint32'); 	% tag, datatype
                        namelen  = fread(HDR.FILE.FID,1,'uint32'); 	% tag, datatype
                        if namelen>HDR.FILE.size,
			%	fclose(HDR.FILE.FID);
				HDR.ERROR.status  = -1; 
				HDR.ERROR.message = sprintf('Error SOPEN (MAT4): Could not open %s\n',HDR.FileName);
				return;
			end;
			[name,c] = fread(HDR.FILE.FID,namelen,'char'); 
                        
                        if imagf, 
				HDR.ERROR.status=-1; 
				fprintf(HDR.FILE.stderr,'Warning %s: Imaginary data not tested\n',mfilename); 
			end;
                        if type(4)==2,
                                HDR.ERROR.status=-1;
                                fprintf(HDR.FILE.stderr,'Error %s: sparse data not supported\n',mfilename);
                        elseif type(4)>2, 
                                type(4)=rem(type(4),2);
                        end;
                        
                        dt=type(3);
                        if     dt==0, SIZOF=8; TYP = 'float64';
                        elseif dt==6, SIZOF=1; TYP = 'uint8';
                        elseif dt==4, SIZOF=2; TYP = 'uint16';
                        elseif dt==3, SIZOF=2; TYP = 'int16';
                        elseif dt==2, SIZOF=4; TYP = 'int32';
                        elseif dt==1, SIZOF=4; TYP = 'float32';
                        else
                                fprintf(HDR.FILE.stderr,'Error %s: unknown data type\n',mfilename);
                        end;
                        
                        HDR.Var(k).Name  = char(name(1:length(name)-1)');
                        HDR.Var(k).Size  = [mrows,ncols];
                        HDR.Var(k).SizeOfType = SIZOF;
                        HDR.Var(k).Type  = [type;~~imagf]';
                        HDR.Var(k).TYP   = TYP;
                        HDR.Var(k).Pos   = ftell(HDR.FILE.FID);
                        
                        c=0; 
                        %% find the ADICHT data channels
                        if strfind(HDR.Var(k).Name,'data_block'),
                                HDR.ADI.DB(str2double(HDR.Var(k).Name(11:length(HDR.Var(k).Name))))=k;
                        elseif strfind(HDR.Var(k).Name,'ticktimes_block'),
                                HDR.ADI.TB(str2double(HDR.Var(k).Name(16:length(HDR.Var(k).Name))))=k;
                        end;
                        
                        tmp1=ftell(HDR.FILE.FID);
                        
                        % skip next block
                        tmp=(prod(HDR.Var(k).Size)-c)*HDR.Var(k).SizeOfType*(1+(~~imagf));
                        fseek(HDR.FILE.FID,tmp,0); 
                        
                        tmp2=ftell(HDR.FILE.FID);
                        if (tmp2-tmp1) < tmp,  % if skipping the block was not successful
                                HDR.ErrNo = -1;
                                HDR.ERROR = sprintf('file %s is corrupted',HDR.FileName);
                                fprintf(HDR.FILE.stderr,'Error SOPEN: MAT4 (ADICHT) file %s is corrupted\n',HDR.FileName);
                                return;
                        end;	                
                        
                        %type = fread(HDR.FILE.FID,4,'uchar');  	% 4-byte header
                        type = fread(HDR.FILE.FID,1,'uint32'); 	% 4-byte header
	        end;
	        HDR.FILE.OPEN = 1;
	        HDR.FILE.POS = 0;
        
	
    	if isfield(HDR,'ADI')
                HDR.TYPE = 'ADI', % ADICHT-data, converted into a Matlab 4 file
    		
	        fprintf(HDR.FILE.stderr,'Format not tested yet. \nFor more information contact <a.schloegl@ieee.org> Subject: Biosig/Dataformats \n',PERMISSION);	

                %% set internal sampling rate to 1000Hz (default). Set HDR.iFs=[] if no resampling should be performed 
                HDR.iFs = []; %1000;
                HDR.NS  = HDR.Var(HDR.ADI.DB(1)).Size(1);
                HDR.ADI.comtick = [];        
                HDR.ADI.comTick = [];        
                HDR.ADI.comtext = [];
                HDR.ADI.comchan = [];
                HDR.ADI.comblok = [];
                HDR.ADI.index   = [];
                HDR.ADI.range   = [];
                HDR.ADI.scale   = [];
                HDR.ADI.titles  = [];
                
                HDR.ADI.units   = [];
                
                for k=1:length(HDR.ADI.TB),
                        [HDR,t1] = matread(HDR,['ticktimes_block' int2str(k)],[1 2]);	% read first and second element of timeblock
                        [HDR,t2] = matread(HDR,['ticktimes_block' int2str(k)],HDR.Var(HDR.ADI.DB(k)).Size(2)); % read last element of timeblock
                        HDR.ADI.ti(k,1:2) = [t1(1),t2];
                        HDR.SampleRate(k) = round(1/diff(t1));
                        
                        [HDR,tmp] = matread(HDR,['comtick_block' int2str(k)]);	% read first and second element of timeblock
                        HDR.ADI.comtick = [HDR.ADI.comtick;tmp];
                        %HDR.ADI.comTick = [HDR.ADI.comTick;tmp/HDR.SampleRate(k)+HDR.ADI.ti(k,1)];
                        [HDR,tmp] = matread(HDR,['comchan_block' int2str(k)]);	% read first and second element of timeblock
                        HDR.ADI.comchan = [HDR.ADI.comchan;tmp];
                        [HDR,tmp] = matread(HDR,['comtext_block' int2str(k)]);	% read first and second element of timeblock
                        tmp2 = size(HDR.ADI.comtext,2)-size(tmp,2);
                        if tmp2>=0,
                                HDR.ADI.comtext = [HDR.ADI.comtext;[tmp,zeros(size(tmp,1),tmp2)]];
                        else
                                HDR.ADI.comtext = [[HDR.ADI.comtext,zeros(size(HDR.ADI.comtext,1),-tmp2)];tmp];
                        end;
                        HDR.ADI.comblok=[HDR.ADI.comblok;repmat(k,size(tmp,1),1)];
                        
                        [HDR,tmp] = matread(HDR,['index_block' int2str(k)]);	% read first and second element of timeblock
                        if isempty(tmp),
                                HDR.ADI.index{k} = 1:HDR.NS;
                        else
                                HDR.NS=length(tmp); %
                                HDR.ADI.index{k} = tmp;
                        end;
                        [HDR,tmp] = matread(HDR,['range_block' int2str(k)]);	% read first and second element of timeblock
                        HDR.ADI.range{k} = tmp;
                        [HDR,tmp] = matread(HDR,['scale_block' int2str(k)]);	% read first and second element of timeblock
                        HDR.ADI.scale{k} = tmp;
                        [HDR,tmp] = matread(HDR,['titles_block' int2str(k)]);	% read first and second element of timeblock
                        HDR.ADI.titles{k} = tmp;
                        
                        [HDR,tmp] = matread(HDR,['units_block' int2str(k)]);	% read first and second element of timeblock
                        HDR.ADI.units{k} = char(tmp);
                        if k==1;
                                HDR.PhysDim = char(sparse(find(HDR.ADI.index{1}),1:sum(HDR.ADI.index{1}>0),1)*HDR.ADI.units{1}); % for compatibility with the EDF toolbox
                        elseif any(size(HDR.ADI.units{k-1})~=size(tmp))
                                fprintf(HDR.FILE.stderr,'Warning MATOPEN: Units are different from block to block\n');
                        elseif any(any(HDR.ADI.units{k-1}~=tmp))
                                fprintf(HDR.FILE.stderr,'Warning MATOPEN: Units are different from block to block\n');
                        end;	
                        HDR.PhysDim = char(sparse(find(HDR.ADI.index{k}),1:sum(HDR.ADI.index{k}>0),1)*HDR.ADI.units{k}); % for compatibility with the EDF toolbox
                        %HDR.PhysDim=HDR.ADI.PhysDim;
                end;                
                HDR.T0 = datevec(datenum(1970,1,1)+HDR.ADI.ti(1,1)/24/3600);
                for k=1:size(HDR.ADI.comtext,1),
                        HDR.ADI.comtime0(k)=HDR.ADI.comtick(k)./HDR.SampleRate(HDR.ADI.comblok(k))'+HDR.ADI.ti(HDR.ADI.comblok(k),1)-HDR.ADI.ti(1,1);
                end;	
                
                % Test if timeindex is increasing
                tmp = size(HDR.ADI.ti,1);
                if ~all(HDR.ADI.ti(2:tmp,2)>HDR.ADI.ti(1:tmp-1,1)), 
                        HDR.ErrNo=-1;
                        fprintf(HDR.FILE.stderr,'Warning MATOPEN: Time index are not monotonic increasing !!!\n');
                        return;
                end;	
                % end of ADI-Mode
	        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,ones(1,HDR.NS));
        else        
                fclose(HDR.FILE.FID);
                HDR.FILE.FID = -1;
		return; 
        end;
        
        
elseif strcmp(HDR.TYPE,'BCI2003_Ia+b');
        % BCI competition 2003, dataset 1a+b (Tuebingen)
        data = load('-ascii',HDR.FileName);
        if strfind(HDR.FileName,'Testdata'),
                HDR.Classlabel = repmat(NaN,size(data,1),1);
        else
                HDR.Classlabel = data(:,1);
                data = data(:,2:end);
        end;
        
        HDR.NRec = length(HDR.Classlabel);
        HDR.FLAG.TRIGGERED = HDR.NRec>1; 
        HDR.PhysDim = 'µV';
        HDR.SampleRate = 256; 
        
        if strfind(HDR.FILE.Path,'a34lkt') 
                HDR.INFO='BCI competition 2003, dataset 1a (Tuebingen)';
                HDR.Dur = 3.5; 
                HDR.Label = {'A1-Cz';'A2-Cz';'C3f';'C3p';'C4f';'C4p'};
                HDR.TriggerOffset = -2; %[s]
        end;
        
        if strfind(HDR.FILE.Path,'egl2ln')
                HDR.INFO='BCI competition 2003, dataset 1b (Tuebingen)';
                HDR.Dur = 4.5; 
                HDR.Label = {'A1-Cz';'A2-Cz';'C3f';'C3p';'vEOG';'C4f';'C4p'};
                HDR.TriggerOffset = -2; %[s]
        end;
        HDR.SPR = HDR.SampleRate*HDR.Dur;
        HDR.NS  = length(HDR.Label);
        HDR.data = reshape(permute(reshape(data, [HDR.NRec, HDR.SPR, HDR.NS]),[2,1,3]),[HDR.SPR*HDR.NRec,HDR.NS]);
        HDR.TYPE = 'native'; 
        HDR.FILE.POS = 0; 
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
        
        
elseif strcmp(HDR.TYPE,'BCI2003_III');
        % BCI competition 2003, dataset III (Graz)
        tmp = load(HDR.FileName);
        HDR.data = tmp*50;
        if strcmp(HDR.FILE.Name,'x_train'),
                tmp = fullfile(HDR.FILE.Path,'y_train');
                if exist(tmp,'file')
                        HDR.Classlabel = load(tmp);
                end;
        elseif strcmp(HDR.FILE.Name,'x_test'),
                HDR.Classlabel = repmat(NaN,140,1);        
        end;
                
        %elseif isfield(tmp,'x_train') & isfield(tmp,'y_train') & isfield(tmp,'x_test');	
        HDR.INFO  = 'BCI competition 2003, dataset 3 (Graz)'; 
        HDR.Label = {'C3a-C3p'; 'Cza-Czp'; 'C4a-C4p'};
        HDR.SampleRate = 128; 
        HDR.NRec = length(HDR.Classlabel);
        HDR.FLAG.TRIGGERED = 1; 
        HDR.Dur = 9; 
        HDR.NS  = 3;
        HDR.SPR = HDR.SampleRate*HDR.Dur;
        
        sz = [HDR.NS, HDR.SPR, HDR.NRec];
        HDR.data = reshape(permute(HDR.data,[2,1,3]),sz(1),sz(2)*sz(3))';
        HDR.TYPE = 'native'; 
        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
                
                
elseif strncmp(HDR.TYPE,'MAT',3),
        status = warning;
        warning('off');
        tmp = load('-mat',HDR.FileName);
        warning(status);
        if isfield(tmp,'HDR')
                HDR = tmp.HDR; 
                if isfield(HDR,'data');
                        HDR.TYPE = 'native'; 
                end; 
                
        elseif isfield(tmp,'y'),		% Guger, Mueller, Scherer
                HDR.NS = size(tmp.y,2);
                HDR.NRec = 1; 
                if ~isfield(tmp,'SampleRate')
                        %fprintf(HDR.FILE.stderr,['Samplerate not known in ',HDR.FileName,'. 125Hz is chosen']);
                        HDR.SampleRate=125;
                else
                        HDR.SampleRate=tmp.SampleRate;
                end;
                fprintf(HDR.FILE.stderr,'Sensitivity not known in %s.\n',HDR.FileName);
                HDR.data = tmp.y;
                HDR.TYPE = 'native'; 
                
                
        elseif [isfield(tmp,'cnt') | isfield(tmp,'X') ] & isfield(tmp,'nfo')
        	if isfield(tmp,'cnt') 
                        HDR.data = tmp.cnt;
                        [HDR.SPR,HDR.NS] = size(tmp.cnt);
			HDR.INFO='BCI competition 2005, dataset IV (Berlin)'; 
			HDR.Filter.LowPass = 0.05; 
			HDR.Filter.HighPass = 200; 
			HDR.Cal   = 0.1; 
			HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,.1);
		elseif isfield(tmp,'X'),
                        HDR.data = tmp.X;
			[HDR.SPR,HDR.NS] = size(tmp.X);
			HDR.INFO='BCI competition 2005, dataset V (IDIAP)'; 
			HDR.Filter.LowPass = 0; 
			HDR.Filter.HighPass = 256; 
			if isfield(tmp,'Y'),
				HDR.Classlabel = tmp.Y(:);
                        else
				HDR.Classlabel = repmat(NaN,size(tmp.X,1),1);
			end;	
			HDR.Cal   = 1; 
			HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
		else
		
		end;
                
		HDR.PhysDim = 'uV';
		HDR.SampleRate = tmp.nfo.fs; 
		%HDR.Dur = HDR.SPR/HDR.SampleRate;
		if isfield(tmp,'mrk')
			HDR.TRIG  = tmp.mrk.pos; 
			HDR.EVENT.POS = tmp.mrk.pos(:); 
			HDR.EVENT.TYP = zeros(size(HDR.EVENT.POS));
			HDR.EVENT.CHN = zeros(size(HDR.EVENT.POS));
                        if ~isempty(strfind(HDR.INFO,'Berlin')),cuelen=3.5; 
                        elseif ~isempty(strfind(HDR.INFO,'IDIAP')),cuelen=20; 
                        end;
			HDR.EVENT.DUR = repmat(cuelen*HDR.SampleRate,size(HDR.EVENT.POS));
			if isfield(tmp.mrk,'y'),
				HDR.Classlabel = tmp.mrk.y; 
			else	
				HDR.Classlabel = repmat(NaN,size(HDR.TRIG));
			end;
			if isfield(tmp.mrk,'className'),
				HDR.EVENT.TeegType = tmp.mrk.className;
                                HDR.EVENT.TYP(isnan(HDR.Classlabel)) = hex2dec('030f');  % unknown/undefined
				ix = strmatch('left',tmp.mrk.className); 
				if ~isempty(ix),
					HDR.EVENT.TYP(HDR.Classlabel==ix) = hex2dec('0301');  % left
				end;	
				ix = strmatch('right',tmp.mrk.className); 
				if ~isempty(ix),
					HDR.EVENT.TYP(HDR.Classlabel==ix) = hex2dec('0302');  % right
				end;	
				ix = strmatch('foot',tmp.mrk.className); 
				if ~isempty(ix),
					HDR.EVENT.TYP(HDR.Classlabel==ix) = hex2dec('0303');  % foot
				end;	
				ix = strmatch('tongue',tmp.mrk.className); 
				if ~isempty(ix),
					HDR.EVENT.TYP(HDR.Classlabel==ix) = hex2dec('0304');  % tongue
				end;	
			end;
		end;
		HDR.Label = tmp.nfo.clab';
		z2=sum([tmp.nfo.xpos,tmp.nfo.ypos].^2,2);
		HDR.ELEC.XYZ = [tmp.nfo.xpos,tmp.nfo.ypos,sqrt(max(z2)-sz)];
                HDR.NRec = 1; 
		HDR.FILE.POS = 0; 
                HDR.TYPE = 'native'; 
                clear tmp; 
		
                
        elseif isfield(tmp,'Signal') & isfield(tmp,'Flashing') & isfield(tmp,'StimulusCode')
                HDR.INFO = 'BCI competition 2005, dataset II (Albany)'; 
                HDR.SampleRate = 240; 
		HDR.Filter.LowPass   = 60;
		HDR.Filter.HighPass  = 0.1;
                [HDR.NRec,HDR.SPR,HDR.NS] = size(tmp.Signal); 
		HDR.BCI2000.Flashing = tmp.Flashing;
		HDR.BCI2000.StimulusCode = tmp.StimulusCode;
		if isfield(tmp,'TargetChar')
			HDR.BCI2000.TargetChar = tmp.TargetChar;
		end;	
		if isfield(tmp,'StimulusType')
			HDR.BCI2000.StimulusType = tmp.StimulusType;
		end;	

		HDR.FILE.POS = 0; 
                HDR.TYPE = 'native'; 
		HDR.data = reshape(tmp.Signal,[HDR.NRec*HDR.SPR, HDR.NS]);
		clear tmp;
		
                
        elseif isfield(tmp,'run') & isfield(tmp,'trial') & isfield(tmp,'sample') & isfield(tmp,'signal') & isfield(tmp,'TargetCode');
                HDR.INFO='BCI competition 2003, dataset 2a (Albany)'; 
                HDR.SampleRate = 160; 
                HDR.NRec = 1; 
		[HDR.SPR,HDR.NS]=size(tmp.signal);
                HDR.data = tmp.signal; 
                HDR.EVENT.POS = [0;find(diff(tmp.trial)>0)-1];
                HDR.EVENT.TYP = ones(length(HDR.EVENT.POS),1)*hex2dec('0300'); % trial onset; 
                
                if 0,
                        EVENT.POS = [find(diff(tmp.trial)>0);length(tmp.trial)];
                        EVENT.TYP = ones(length(EVENT.POS),1)*hex2dec('8300'); % trial offset; 
                        HDR.EVENT.POS = [HDR.EVENT.POS; EVENT.POS];
                        HDR.EVENT.TYP = [HDR.EVENT.TYP; EVENT.TYP];
                        [HDR.EVENT.POS,ix]=sort(HDR.EVENT.POS);
                        HDR.EVENT.TYP = HDR.EVENT.TYP(ix);
                end;
                
                HDR.EVENT.N = length(HDR.EVENT.POS);
                ix = find((tmp.TargetCode(1:end-1)==0) & (tmp.TargetCode(2:end)>0));
                HDR.Classlabel = tmp.TargetCode(ix+1); 
                HDR.TYPE = 'native'; 

                
        elseif isfield(tmp,'runnr') & isfield(tmp,'trialnr') & isfield(tmp,'samplenr') & isfield(tmp,'signal') & isfield(tmp,'StimulusCode');
                HDR.INFO = 'BCI competition 2003, dataset 2b (Albany)'; 
                HDR.SampleRate = 240; 
                HDR.NRec = 1; 
		[HDR.SPR,HDR.NS]=size(tmp.signal);
                HDR.data = tmp.signal; 
                HDR.EVENT.POS = [0;find(diff(tmp.trialnr)>0)-1];
                HDR.EVENT.TYP = ones(length(HDR.EVENT.POS),1)*hex2dec('0300'); % trial onset; 

                if 0,
                        EVENT.POS = [find(diff(tmp.trial)>0);length(tmp.trial)];
                        EVENT.TYP = ones(length(EVENT.POS),1)*hex2dec('8300'); % trial offset; 
                        HDR.EVENT.POS = [HDR.EVENT.POS; EVENT.POS];
                        HDR.EVENT.TYP = [HDR.EVENT.TYP; EVENT.TYP];
                        [HDR.EVENT.POS,ix]=sort(HDR.EVENT.POS);
                        HDR.EVENT.TYP = HDR.EVENT.TYP(ix);
                end;
                
                HDR.EVENT.N = length(HDR.EVENT.POS);
                ix = find((tmp.StimulusCode(1:end-1)==0) & (tmp.StimulusCode(2:end)>0));
                HDR.Classlabel = tmp.StimulusCode(ix+1); 
                HDR.TYPE = 'native'; 
                
                
        elseif isfield(tmp,'clab') & isfield(tmp,'x_train') & isfield(tmp,'y_train') & isfield(tmp,'x_test');	
                HDR.INFO  = 'BCI competition 2003, dataset 4 (Berlin)'; 
                HDR.Label = tmp.clab;        
                HDR.Classlabel = [repmat(nan,size(tmp.x_test,3),1);tmp.y_train';repmat(nan,size(tmp.x_test,3),1)];
                HDR.NRec  = length(HDR.Classlabel);
                
                HDR.SampleRate = 1000;
                HDR.Dur = 0.5; 
                HDR.NS  = size(tmp.x_test,2);
                HDR.SPR = HDR.SampleRate*HDR.Dur;
                HDR.FLAG.TRIGGERED = 1; 
                sz = [HDR.NS,HDR.SPR,HDR.NRec];
                
                HDR.data = reshape(permute(cat(3,tmp.x_test,tmp.x_train,tmp.x_test),[2,1,3]),sz(1),sz(2)*sz(3))';
                HDR.TYPE = 'native'; 
                
       elseif isfield(tmp,'x_train') & isfield(tmp,'y_train') & isfield(tmp,'x_test');	
                HDR.INFO  = 'BCI competition 2003, dataset 3 (Graz)'; 
                HDR.Label = {'C3a-C3p'; 'Cza-Czp'; 'C4a-C4p'};
                HDR.SampleRate = 128; 
                HDR.Classlabel = [tmp.y_train-1; repmat(nan,size(tmp.x_test,3),1)];
                HDR.data = cat(3, tmp.x_test, tmp.x_train)*50;
                
                HDR.NRec = length(HDR.Classlabel);
                HDR.FLAG.TRIGGERED = 1; 
                HDR.SampleRate = 128;
                HDR.Dur = 9; 
                HDR.NS  = 3;
                HDR.SPR = HDR.SampleRate*HDR.Dur;
                
                sz = [HDR.NS, HDR.SPR, HDR.NRec];
                HDR.data = reshape(permute(HDR.data,[2,1,3]),sz(1),sz(2)*sz(3))';
                HDR.TYPE = 'native'; 
                
                
        elseif isfield(tmp,'RAW_SIGNALS')    % TFM Matlab export 
                HDR.Label = fieldnames(tmp.RAW_SIGNALS);
                HDR.NS = length(HDR.Label); 
                HDR.SampleRate = 1000; 
		ix  = repmat(NaN,1,HDR.NS); 
                for k1 = 1:HDR.NS;
                        s = getfield(tmp.RAW_SIGNALS,HDR.Label{k1});
                        for k2 = 1:length(s);
                                ix(k2,k1) = length(s{k2});   
                        end;
                end;
		DIV = sum(ix,1);
                HDR.TFM.DIV = round(max(DIV)./DIV);
		
                HDR.data = repmat(NaN, max(HDR.TFM.DIV.*DIV), HDR.NS);
                for k1 = 1:HDR.NS;
                        s = getfield(tmp.RAW_SIGNALS,HDR.Label{k1});
                        s2= rs(cat(2,s{:})',1,HDR.TFM.DIV(k1));
			HDR.data(1:size(s2,1),k1) = s2; 
                end;
		clear tmp s s2; 
		HDR.EVENT.POS = cumsum(ix(:,min(find(HDR.TFM.DIV==1)))); 
		HDR.EVENT.TYP = repmat(1,size(HDR.EVENT.POS)); 
                HDR.TFM.SampleRate = HDR.SampleRate./HDR.TFM.DIV; 
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1); 
                HDR.TYPE  = 'native'; 
                HDR.NRec  = 1; 

                
        elseif isfield(tmp,'eeg');	% Scherer
                fprintf(HDR.FILE.stderr,'Warning SLOAD: Sensitivity not known in %s,\n',HDR.FileName);
                HDR.NS=size(tmp.eeg,2);
                HDR.NRec = 1; 
                if ~isfield(tmp,'SampleRate')
                        %fprintf(HDR.FILE.stderr,['Samplerate not known in ',HDR.FileName,'. 125Hz is chosen']);
                        HDR.SampleRate=125;
                else
                        HDR.SampleRate=tmp.SampleRate;
                end;
                HDR.data = tmp.eeg;
                if isfield(tmp,'classlabel'),
                	HDR.Classlabel = tmp.classlabel;
                end;        
                HDR.TYPE = 'native'; 

                
        elseif isfield(tmp,'data');
                if isfield(tmp,'readme') & iscell(tmp.data) ;	%Zachary A. Keirn, Purdue University, 1988. 
                        HDR.Label = {'C3'; 'C4'; 'P3'; 'P4'; 'O1'; 'O2'; 'EOG'};                               
                        HDR.SampleRate = 250; 
                        HDR.FLAG.TRIGGERED  = 1; 
                        HDR.DUR = 10; 
                        HDR.SPR = 2500;
                        HDR.FILTER.LowPass  = 0.1;
                        HDR.FILTER.HighPass = 100; 
                        HDR.NRec = length(tmp.data);
                        
                        x = cat(1,tmp.data{:});
                        [b,i,CL] = unique({x{:,1}}');
                        [HDR.EVENT.TeegDesc,i,CL(:,2)] = unique({x{:,2}}');
                        HDR.Classlabel = CL; 
                        HDR.data = [x{:,4}]';
                        HDR.NS   = size(HDR.data,2); 
                        HDR.Calib= sparse(2:8,1:7,1);
                        HDR.TYPE = 'native'; 
                        
                else        	% Mueller, Scherer ? 
                        HDR.NS = size(tmp.data,2);
                        HDR.NRec = 1; 
                        fprintf(HDR.FILE.stderr,'Warning SLOAD: Sensitivity not known in %s,\n',HDR.FileName);
                        if ~isfield(tmp,'SampleRate')
                                fprintf(HDR.FILE.stderr,'Warning SLOAD: Samplerate not known in %s. 125Hz is chosen\n',HDR.FileName);
                                HDR.SampleRate=125;
                        else
                                HDR.SampleRate=tmp.SampleRate;
                        end;
                        HDR.data = tmp.data;
                        if isfield(tmp,'classlabel'),
                                HDR.Classlabel = tmp.classlabel;
                        end;        
                        if isfield(tmp,'artifact'),
                                HDR.ArtifactSelection = zeros(size(tmp.classlabel));
                                HDR.ArtifactSelection(tmp.artifact)=1;
                        end;        
                        HDR.TYPE = 'native'; 
                end;
                
                
        elseif isfield(tmp,'EEGdata');  % Telemonitoring Daten (Reinhold Scherer)
                HDR.NS = size(tmp.EEGdata,2);
                HDR.NRec = 1; 
                HDR.Classlabel = tmp.classlabel;
                if ~isfield(tmp,'SampleRate')
                        fprintf(HDR.FILE.stderr,'Warning SLOAD: Samplerate not known in %s. 125Hz is chosen\n',HDR.FileName);
                        HDR.SampleRate=125;
                else
                        HDR.SampleRate=tmp.SampleRate;
                end;
                HDR.PhysDim = 'µV';
                fprintf(HDR.FILE.stderr,'Sensitivity not known in %s. 50µV is chosen\n',HDR.FileName);
                        HDR.data = tmp.EEGdata*50;
                HDR.TYPE = 'native'; 
                
        elseif isfield(tmp,'daten');	% EP Daten von Michael Woertz
                HDR.NS = size(tmp.daten.raw,2)-1;
                HDR.NRec = 1; 
                if ~isfield(tmp,'SampleRate')
                        fprintf(HDR.FILE.stderr,'Warning SLOAD: Samplerate not known in %s. 2000Hz is chosen\n',HDR.FileName);
                        HDR.SampleRate=2000;
                else
                        HDR.SampleRate=tmp.SampleRate;
                end;
                HDR.PhysDim = 'µV';
                fprintf(HDR.FILE.stderr,'Sensitivity not known in %s. 100µV is chosen\n',HDR.FileName);
                %signal=tmp.daten.raw(:,1:HDR.NS)*100;
                HDR.data = tmp.daten.raw*100;
                HDR.TYPE = 'native'; 
                
        elseif isfield(tmp,'neun') & isfield(tmp,'zehn') & isfield(tmp,'trig');	% guger, 
                HDR.NS=3;
                HDR.NRec = 1; 
                if ~isfield(tmp,'SampleRate')
                        fprintf(HDR.FILE.stderr,'Warning SLOAD: Samplerate not known in %s. 125Hz is chosen\n',HDR.FileName);
                        HDR.SampleRate=125;
                else
                        HDR.SampleRate=tmp.SampleRate;
                end;
                fprintf(HDR.FILE.stderr,'Sensitivity not known in %s. \n',HDR.FileName);
                HDR.data = [tmp.neun;tmp.zehn;tmp.trig];
                HDR.Label = {'Neun','Zehn','TRIG'};
                HDR.TYPE = 'native'; 
                
                
        elseif isfield(tmp,'Recorder1')    % Nicolet NRF format converted into Matlab 
                for k = 1:length(s.Recorder1.Channels.ChannelInfos);
                        HDR.Label{k} = [s.Recorder1.Channels.ChannelInfos(k).ChannelInfo.Name,' '];
                        HDR.PhysDim{k} = [s.Recorder1.Channels.ChannelInfos(k).ChannelInfo.YUnits,' '];
                end;
                signal = [];
                T = [];
                for k = 1:length(s.Recorder1.Channels.Segments)
                        tmp = s.Recorder1.Channels.Segments(k).Data;
                        sz = size(tmp.Samples);
                        signal = [signal; repmat(nan,100,sz(1)); tmp.Samples'];
                        T = [T;repmat(nan,100,1);tmp.dX0+(1:sz(2))'*tmp.dXstep ]
                        fs = 1./tmp.dXstep;
                        if k==1,
                                HDR.SampleRate = fs;
                        elseif HDR.SampleRate ~= fs; 
                                fprintf(2,'Error SLOAD (NRF): different Sampling rates not supported, yet.\n');
                        end;
                end;
                HDR.data = signal; 
                HDR.TYPE = 'native'; 

                
        elseif isfield(tmp,'ECoGdata') & isfield(tmp,'dataset')  %Michigan ECoG dataset 
                HDR.data = tmp.ECoGdata';
                HDR.T0 = datevec(datenum(tmp.dataset.filetype.timestamp));
                HDR.SampleRate = tmp.dataset.specs.sample_rate;
                HDR.Filter.HighPass = tmp.dataset.specs.filters.lowcut;
                HDR.Filter.LowPass = tmp.dataset.specs.filters.highcut;
                if isfield(tmp.dataset.specs.filters,'notch60');
                        HDR.FILTER.Notch = tmp.dataset.specs.filters.notch60*60;
                end;
                HDR.Patient.Sex = tmp.dataset.subject_info.gender; 
                HDR.Patient.Age = tmp.dataset.subject_info.age; 
                HDR.Label = tmp.dataset.electrode.names;
                HDR.NS    = tmp.dataset.electrode.number;

                trigchancode = getfield(tmp.dataset.electrode.options,'TRIGGER');
                HDR.AS.TRIGCHAN = find(tmp.dataset.electrode.region==trigchancode);
                HDR.TRIG  = tmp.dataset.trigger.trigs_all;
                
                HDR.FLAG.TRIGGERED = 0;
                HDR.NRec  = 1; 
                HDR.SPR = size(HDR.data,1);
                HDR.Dur = HDR.SPR/HDR.SampleRate;
                HDR.TYPE  = 'native'; 
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
                clear tmp; 
                
                
        elseif isfield(tmp,'P_C_S');	% G.Tec Ver 1.02, 1.5x data format
                HDR.FILE.POS = 0; 
                if isa(tmp.P_C_S,'data'), %isfield(tmp.P_C_S,'version'); % without BS.analyze	
                        if any(tmp.P_C_S.Version==[1.02, 1.5, 1.52]),
                        else
                                fprintf(HDR.FILE.stderr,'Warning: PCS-Version is %4.2f.\n',tmp.P_C_S.Version);
                        end;
                        HDR.Filter.LowPass  = tmp.P_C_S.LowPass;
                        HDR.Filter.HighPass = tmp.P_C_S.HighPass;
                        HDR.Filter.Notch    = tmp.P_C_S.Notch;
                        HDR.SampleRate      = tmp.P_C_S.SamplingFrequency;
                        HDR.gBS.Attribute   = tmp.P_C_S.Attribute;
                        HDR.gBS.AttributeName = tmp.P_C_S.AttributeName;
                        HDR.Label = tmp.P_C_S.ChannelName;
                        HDR.gBS.EpochingSelect = tmp.P_C_S.EpochingSelect;
                        HDR.gBS.EpochingName = tmp.P_C_S.EpochingName;

                        HDR.data = double(tmp.P_C_S.Data);
                        
                else %if isfield(tmp.P_C_S,'Version'),	% with BS.analyze software, ML6.5
                        if any(tmp.P_C_S.version==[1.02, 1.5, 1.52]),
                        else
                                fprintf(HDR.FILE.stderr,'Warning: PCS-Version is %4.2f.\n',tmp.P_C_S.version);
                        end;        
                        HDR.Filter.LowPass  = tmp.P_C_S.lowpass;
                        HDR.Filter.HighPass = tmp.P_C_S.highpass;
                        HDR.Filter.Notch    = tmp.P_C_S.notch;
                        HDR.SampleRate      = tmp.P_C_S.samplingfrequency;
                        HDR.gBS.Attribute   = tmp.P_C_S.attribute;
                        HDR.gBS.AttributeName = tmp.P_C_S.attributename;
                        HDR.Label = tmp.P_C_S.channelname;
                        HDR.gBS.EpochingSelect = tmp.P_C_S.epochingselect;
                        HDR.gBS.EpochingName = tmp.P_C_S.epochingname;
                        
                        HDR.data = double(tmp.P_C_S.data);
                end;
                tmp = []; % clear memory

                sz     = size(HDR.data);
                HDR.NRec = sz(1);
                HDR.SPR  = sz(2);
                HDR.Dur  = sz(2)/HDR.SampleRate;
                HDR.NS   = sz(3);
                HDR.FLAG.TRIGGERED = HDR.NRec>1;
                
                HDR.data  = reshape(permute(HDR.data,[2,1,3]),[sz(1)*sz(2),sz(3)]);

                % Selection of trials with artifacts
                ch = strmatch('ARTIFACT',HDR.gBS.AttributeName);
                if ~isempty(ch)
                        HDR.ArtifactSelection = HDR.gBS.Attribute(ch,:);
                end;
                
                % Convert gBS-epochings into BIOSIG - Events
                map = zeros(size(HDR.gBS.EpochingName,1),1);
                map(strmatch('AUGE',HDR.gBS.EpochingName))=hex2dec('0101');
                map(strmatch('EOG',HDR.gBS.EpochingName))=hex2dec('0101');
                map(strmatch('MUSKEL',HDR.gBS.EpochingName))=hex2dec('0103');
                map(strmatch('MUSCLE',HDR.gBS.EpochingName))=hex2dec('0103');
                map(strmatch('ELECTRODE',HDR.gBS.EpochingName))=hex2dec('0105');

                map(strmatch('SLEEPSTAGE1',HDR.gBS.EpochingName))=hex2dec('0411');
                map(strmatch('SLEEPSTAGE2',HDR.gBS.EpochingName))=hex2dec('0412');
                map(strmatch('SLEEPSTAGE3',HDR.gBS.EpochingName))=hex2dec('0413');
                map(strmatch('SLEEPSTAGE4',HDR.gBS.EpochingName))=hex2dec('0414');
                map(strmatch('REM',HDR.gBS.EpochingName))=hex2dec('0415');

                if ~isempty(HDR.gBS.EpochingSelect),
                        HDR.EVENT.TYP = map([HDR.gBS.EpochingSelect{:,9}]');
                        HDR.EVENT.POS = [HDR.gBS.EpochingSelect{:,1}]';
                        HDR.EVENT.CHN = [HDR.gBS.EpochingSelect{:,3}]';
                        HDR.EVENT.DUR = [HDR.gBS.EpochingSelect{:,4}]';
                end;
                HDR.TYPE = 'native'; 
                
	elseif isfield(tmp,'P_C_DAQ_S');
                if ~isempty(tmp.P_C_DAQ_S.data),
                        HDR.data = double(tmp.P_C_DAQ_S.data{1});
                        
                elseif ~isempty(tmp.P_C_DAQ_S.daqboard),
                        [tmppfad,file,ext] = fileparts(tmp.P_C_DAQ_S.daqboard{1}.ObjInfo.LogFileName),
                        file = [file,ext];
                        if exist(file,'file')
                                HDR.data=daqread(file);        
                                HDR.info=daqread(file,'info');        
                        else
                                fprintf(HDR.FILE.stderr,'Error SLOAD: no data file found\n');
                                return;
                        end;
                        
                else
                        fprintf(HDR.FILE.stderr,'Error SLOAD: no data file found\n');
                        return;
                end;
                
                HDR.NS = size(HDR.data,2);
                HDR.Cal = tmp.P_C_DAQ_S.sens*(2.^(1-tmp.P_C_DAQ_S.daqboard{1}.HwInfo.Bits));
                HDR.Calib = sparse(2:HDR.NS,1:HDR.NS,HDR.Cal);
                
                if all(tmp.P_C_DAQ_S.unit==1)
                        HDR.PhysDim='uV';
                else
                        HDR.PhysDim='[?]';
                end;
                
                HDR.SampleRate = tmp.P_C_DAQ_S.samplingfrequency;
                sz     = size(HDR.data);
                if length(sz)==2, sz=[1,sz]; end;
                HDR.NRec = sz(1);
                HDR.Dur  = sz(2)/HDR.SampleRate;
                HDR.NS   = sz(3);
                HDR.FLAG.TRIGGERED = HDR.NRec>1;
                HDR.Filter.LowPass = tmp.P_C_DAQ_S.lowpass;
                HDR.Filter.HighPass = tmp.P_C_DAQ_S.highpass;
                HDR.Filter.Notch = tmp.P_C_DAQ_S.notch;
                HDR.TYPE = 'native'; 
                
                
        elseif isfield(tmp,'eventmatrix') & isfield(tmp,'samplerate') 
                %%% F. Einspieler's Event information 
                HDR.EVENT.POS = tmp.eventmatrix(:,1);
                HDR.EVENT.TYP = tmp.eventmatrix(:,2);
                HDR.EVENT.CHN = tmp.eventmatrix(:,3);
                HDR.EVENT.DUR = tmp.eventmatrix(:,4);
                HDR.SampleRate = tmp.samplerate;
                HDR.TYPE = 'EVENT';
                
                
        elseif isfield(tmp,'Electrode') 
        	if isfield(tmp.Electrode,'Theta') & isfield(tmp.Electrode,'Phi')
        		Theta = tmp.Electrode.Theta(:)*pi/180; 
        		Phi   = tmp.Electrode.Phi(:)*pi/180; 
        		HDR.ELEC.XYZ = [ sin(Theta).*cos(Phi), sin(Theta).*sin(Phi),cos(Theta)];
			HDR.Label = tmp.Electrode.Acronym(:);
        		HDR.TYPE = 'ELPOS'; 
        		return;
		end;	

        else 
                HDR.Calib = 1; 
                CHAN = 1; 
        end;
        if strcmp(HDR.TYPE,'native'),
                if ~isfield(HDR,'NS');
                        HDR.NS = size(HDR.data,2);
                end;
                if ~isfield(HDR,'SPR');
                        HDR.SPR = size(HDR.data,1);
                end;
                if ~isfield(HDR,'Calib');
                        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
                end;
                if ~isfield(HDR.FILE,'POS');
                        HDR.FILE.POS = 0;
                end;
        end;

        
elseif strncmp(HDR.TYPE,'BCI2000',7),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
                
                [HDR.Header,count] = fread(HDR.FILE.FID,[1,256],'char');
		[tmp,rr] = strtok(char(HDR.Header),[10,13]);
                tmp(tmp=='=') = ' ';
                [t,status,sa] = str2double(tmp,[9,32],[10,13]);
                if (HDR.VERSION==1) & strcmp(sa{3},'SourceCh') & strcmp(sa{5},'StatevectorLen') & ~any(status([2,4,6]))
                        HDR.HeadLen = t(2);
                        HDR.NS = t(4);
                        HDR.BCI2000.StateVectorLength = t(6);
                        HDR.GDFTYP = 'int16';
                elseif (HDR.VERSION==1.1) & strcmp(sa{5},'SourceCh') & strcmp(sa{7},'StatevectorLen') & strcmp(sa{9},'DataFormat') & ~any(status([2:2:8]))
                        HDR.VERSION = t(2);
                        HDR.HeadLen = t(4);
                        HDR.NS = t(6);
                        HDR.BCI2000.StateVectorLength = t(8);
                        HDR.GDFTYP = sa{10};
                else
                        HDR.TYPE = 'unknown'; 
                        fprintf(HDR.FILE.stderr,'Error SOPEN: file %s does not confirm with BCI2000 format\n',HDR.FileName);
                        fclose(HDR.FILE.FID);
                        return; 
                end;
                if count<HDR.HeadLen,
                        status = fseek(HDR.FILE.FID,0,'bof');
                        [HDR.Header,count] = fread(HDR.FILE.FID,[1,HDR.HeadLen],'char');
                elseif count>HDR.HeadLen,
                        status = fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
                        HDR.Header = HDR.Header(1:HDR.HeadLen);
                end
		[tline,rr] = strtok(char(HDR.Header),[10,13]);

                STATUSFLAG = 0;
		while length(rr), 
			tline = tline(1:min([length(tline),strfind(tline,[47,47])-1]));

			if ~isempty(strfind(tline,'[ State Vector Definition ]'))
				STATUSFLAG = 1;
				STATECOUNT = 0; 

			elseif ~isempty(strfind(tline,'[ Parameter Definition ]'))
				STATUSFLAG = 2;

			elseif strncmp(tline,'[',1)
				STATUSFLAG = 3;
			
			elseif STATUSFLAG==1, 
				[t,r] = strtok(tline);
				val = str2double(r);
				%HDR.BCI2000 = setfield(HDR.BCI2000,t,val);
				STATECOUNT = STATECOUNT + 1; 
				HDR.BCI2000.StateVector(STATECOUNT,:) = val; 
				HDR.BCI2000.StateDef{STATECOUNT,1} = t; 
		    
			elseif STATUSFLAG==2, 
				[tag,r] = strtok(tline,'=');
				[val,r] = strtok(r,'=');
				if ~isempty(strfind(tag,'SamplingRate'))
					[tmp,status] = str2double(val);
					HDR.SampleRate = tmp(1);
				elseif ~isempty(strfind(tag,'SourceChGain'))
					[tmp,status] = str2double(val);
					HDR.Cal = tmp(2:tmp(1)+1);
				elseif ~isempty(strfind(tag,'SourceChOffset'))
					[tmp,status] = str2double(val);
					HDR.Off = tmp(2:tmp(1)+1);
				elseif ~isempty(strfind(tag,'SourceMin'))
					[tmp,status] = str2double(val);
					HDR.DigMin = tmp(1);
				elseif ~isempty(strfind(tag,'SourceMax'))
					[tmp,status] = str2double(val);
					HDR.DigMax = tmp(1);
				end;
			end;	
			[tline,rr] = strtok(rr,[10,13]);
		end;
                HDR.PhysDim = 'µV';
                HDR.Calib = [HDR.Off(1)*ones(1,HDR.NS);eye(HDR.NS)]*HDR.Cal(1);
                
		% decode State Vector Definition 
		X = repmat(NaN,1,HDR.BCI2000.StateVectorLength*8);
		for k = 1:STATECOUNT,
			for k1 = 1:HDR.BCI2000.StateVector(k,1),
				X(HDR.BCI2000.StateVector(k,3:4)*[8;1]+k1) = k;
			end;		
		end;
		%HDR.BCI2000.X = X;
                
		% convert EVENT information
		status = fseek(HDR.FILE.FID,HDR.HeadLen+2*HDR.NS,'bof');
		tmp = fread(HDR.FILE.FID,[HDR.BCI2000.StateVectorLength,inf],[int2str(HDR.BCI2000.StateVectorLength),'*uchar'],HDR.NS*2)';
		HDR.EVENT.POS = [1;1+find(any(diff(tmp,[],1),2))];
		HDR.EVENT.DUR = diff([HDR.EVENT.POS; 1+size(tmp,1)]);
		HDR.EVENT.CHN = zeros(size(HDR.EVENT.POS));
		HDR.EVENT.TYP = repmat(0,size(HDR.EVENT.POS)); 	% should be extracted from HDR.BCI2000.STATE
		fprintf(2,'Warning SOPEN (BCI2000): HDR.EVENT.TYP information need to be extracted from HDR.BCI2000.STATE\n');

                tmp = tmp(HDR.EVENT.POS,:)';         % compress event information
                HDR.BCI2000.BINARYSTATUS = reshape(dec2bin(tmp(:),8)',8*HDR.BCI2000.StateVectorLength,size(tmp,2))';
		for  k = 1:max(X)
                        HDR.BCI2000.STATE(:,k) = bin2dec(HDR.BCI2000.BINARYSTATUS(:,k==X));
                end;

		% finalize header definition 		
		status = fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
		HDR.AS.bpb = 2*HDR.NS + HDR.BCI2000.StateVectorLength;
		HDR.SPR    = (HDR.FILE.size - HDR.HeadLen)/HDR.AS.bpb;
		HDR.AS.endpos = HDR.SPR;
		HDR.GDFTYP    = [int2str(HDR.NS),'*',HDR.GDFTYP,'=>',HDR.GDFTYP];
		HDR.NRec      = 1; 
		
                HDR.FILE.OPEN = 1;
		HDR.FILE.POS = 0; 
        end;

        
elseif strcmp(HDR.TYPE,'CFWB'),		% Chart For Windows Binary data, defined by ADInstruments. 
        CHANNEL_TITLE_LEN = 32;
        UNITS_LEN = 32;
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
                
                fprintf(HDR.FILE.stderr,'Format not tested yet. \nFor more information contact <a.schloegl@ieee.org> Subject: Biosig/Dataformats \n',PERMISSION);	
                
                HDR.FILE.OPEN = 1;
                fseek(HDR.FILE.FID,4,'bof');
                HDR.VERSION = fread(HDR.FILE.FID,1,'int32');
                HDR.Dur = fread(HDR.FILE.FID,1,'double');
                HDR.SampleRate = 1/HDR.Dur;
                HDR.T0 = fread(HDR.FILE.FID,5,'int32');
                tmp = fread(HDR.FILE.FID,2,'double');
                HDR.T0(6) = tmp(1);
                HDR.CFWB.preTrigger = tmp(2);
                HDR.NS = fread(HDR.FILE.FID,1,'int32');
                HDR.SPR = fread(HDR.FILE.FID,1,'int32');
                HDR.NRec = 1;
                HDR.FLAG.TRIGGERED = 0;	        
                
                HDR.FLAG.TimeChannel = fread(HDR.FILE.FID,1,'int32');
                tmp = fread(HDR.FILE.FID,1,'int32');
                if tmp == 1, 
                        HDR.GDFTYP = 'float64';
                        HDR.AS.bpb = HDR.NS * 8;
                elseif tmp == 2, 
                        HDR.GDFTYP = 'float32';
                        HDR.AS.bpb = HDR.NS * 4;
                elseif tmp == 3, 
                        HDR.GDFTYP = 'int16';
                        HDR.AS.bpb = HDR.NS * 2;
                end;
                for k = 1:HDR.NS,
                        HDR.Label(k,:) = fread(HDR.FILE.FID,[1, CHANNEL_TITLE_LEN],'char');   
                        HDR.PhysDim(k,:) = fread(HDR.FILE.FID,[1, UNITS_LEN],'char');   
                        HDR.Cal(k,1) = fread(HDR.FILE.FID,1,'double');   
                        HDR.Off(k,1) = fread(HDR.FILE.FID,1,'double');   
                        HDR.PhysMax(1,k) = fread(HDR.FILE.FID,1,'double');   
                        HDR.PhysMin(1,k) = fread(HDR.FILE.FID,1,'double');   
                end;

                
        elseif any(PERMISSION=='w'),
                HDR.VERSION   = 1;
                if ~isfield(HDR,'NS'),
                        HDR.NS = 0; 	% unknown channel number ...
                        fprintf(HDR.FILE.stderr,'Error SOPEN-W CFWB: number of channels HDR.NS undefined.\n');
                        return;
                end;
                if ~isfield(HDR,'SPR'),
                        HDR.SPR = 0; 	% Unknown - Value will be fixed when file is closed. 
                else
                        HDR.SPR = HDR.SPR(1);
                end;
                if ~isfield(HDR,'SampleRate'),
                        HDR.SampleRate = 1; 	% Unknown - Value will be fixed when file is closed. 
                        fprintf(HDR.FILE.stderr,'Warning SOPEN-W CFWB: samplerate undefined.\n');
                end;
                if any([HDR.SPR==0]), 	% if any unknown, ...				HDR.FILE.OPEN = 3;			%	... fix header when file is closed. 
                end;
                if ~isfield(HDR,'CFWB'),
                        HDR.CFWB.preTrigger = 0; 	% Unknown - Value will be fixed when file is closed. 
                end;
                if ~isfield(HDR.CFWB,'preTrigger'),
                        HDR.CFWB.preTrigger = 0; 	% Unknown - Value will be fixed when file is closed. 
                end;
                if ~isfield(HDR,'FLAG'),
                        HDR.FLAG.TimeChannel = 0;
                else
                        if ~isfield(HDR.FLAG,'TimeChannel'),
                                HDR.Flag.TimeChannel = 0;
                        end;
                end;
                if strcmp(gdfdatatype(HDR.GDFTYP),'float64');
                        tmp = 1;
                        HDR.AS.bpb = HDR.NS * 8;
                        HDR.Cal = ones(HDR.NS,1);
                        HDR.Off = zeros(HDR.NS,1);
                elseif strcmp(gdfdatatype(HDR.GDFTYP),'float32');
                        tmp = 2; 
                        HDR.AS.bpb = HDR.NS * 4;
                        HDR.Cal = ones(HDR.NS,1);
                        HDR.Off = zeros(HDR.NS,1);
                elseif strcmp(gdfdatatype(HDR.GDFTYP),'int16');
                        tmp = 3;
                        HDR.AS.bpb = HDR.NS * 2;
                end;
                HDR.PhysMax = repmat(NaN,HDR.NS,1);
                HDR.PhysMin = repmat(NaN,HDR.NS,1);
                if ~isfield(HDR,'Cal'),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN-W CFWB: undefined scaling factor\n');			
                        HDR.Cal = ones(HDR.NS,1);
                end;
                if ~isfield(HDR,'Off'),
                        fprintf(HDR.FILE.stderr,'Warning SOPEN-W CFWB: undefined offset\n');			
                        HDR.Off = zeros(HDR.NS,1);
                end;
                if ~isfield(HDR,'Label'),
                        for k = 1:HDR.NS,
                                Label{k} = sprintf('channel %i',k);
                        end;
                        HDR.Label = strvcat(Label);
                elseif iscell(HDR.Label)
                        for k = 1:min(HDR.NS,length(HDR.Label)),
                                HDR.Label{k}= [HDR.Label{k},' ']; 
                        end;
                        HDR.Label = strvcat(HDR.Label);
                end;
                HDR.Label = [HDR.Label,char(repmat(32,size(HDR.Label,1),max(0,CHANNEL_TITLE_LEN-size(HDR.Label,2))))];
                HDR.Label = [HDR.Label;char(repmat(32,max(0,HDR.NS-size(HDR.Label,1)),size(HDR.Label,2)))];
                
                if ~isfield(HDR,'PhysDim'),
                        HDR.PhysDim = char(repmat(32,HDR.NS,UNITS_LEN));
                end;
                
                if size(HDR.PhysDim,1)==1,
                        HDR.PhysDim = HDR.PhysDim(ones(HDR.NS,1),:);
                end;		
                if iscell(HDR.PhysDim)
                        for k = 1:length(HDR.PhysDim),
                                HDR.PhysDim{k} = [HDR.PhysDim{k},' ']; 
                        end;
                        HDR.PhysDim = strvcat(HDR.PhysDim);
                end
                HDR.PhysDim = [HDR.PhysDim, setstr(repmat(32,size(HDR.PhysDim,1),max(0,UNITS_LEN-size(HDR.PhysDim,2))))];
                HDR.PhysDim = [HDR.PhysDim; setstr(repmat(32,max(0,HDR.NS-size(HDR.PhysDim,1)),size(HDR.PhysDim,2)))];
                
                
                %%%%% write fixed header
                HDR.FILE.FID = fopen(HDR.FileName,PERMISSION,'ieee-le');
                if HDR.FILE.FID<0, 
                        fprintf(HDR.FILE.stderr,'Error SOPEN-W CFWB: could not open file %s .\n',HDR.FileName);
                        return;
                else
                        HDR.FILE.OPEN = 2;		
                end;
                fwrite(HDR.FILE.FID,'CFWB','char');
                fwrite(HDR.FILE.FID,HDR.VERSION,'int32');
                fwrite(HDR.FILE.FID,1/HDR.SampleRate(1),'double');
                fwrite(HDR.FILE.FID,HDR.T0(1:5),'int32');
                fwrite(HDR.FILE.FID,HDR.T0(6),'double');
                fwrite(HDR.FILE.FID,HDR.preTrigger,'double');
                fwrite(HDR.FILE.FID,[HDR.NS,HDR.SPR,HDR.Flag.TimeChannel],'int32');
                fwrite(HDR.FILE.FID,tmp,'int32');
                HDR.HeadLen = ftell(HDR.FILE.FID);
                if (HDR.HeadLen~=68),
                        fprintf(HDR.FILE.stderr,'Error SOPEN CFWB: size of header1 does not fit in file %s\n',HDR.FileName);
                end;
                
                %%%%% write channel header
                for k = 1:HDR.NS,
                        fwrite(HDR.FILE.FID,HDR.Label(k,1:32),'char');
                        fwrite(HDR.FILE.FID,setstr(HDR.PhysDim(k,1:32)),'char');
                        fwrite(HDR.FILE.FID,[HDR.Cal(k),HDR.Off(k)],'double');
                        fwrite(HDR.FILE.FID,[HDR.PhysMax(k),HDR.PhysMin(k)],'double');
                end;
                %HDR.HeadLen = (68+HDR.NS*96); %
                HDR.HeadLen = ftell(HDR.FILE.FID);
                if (HDR.HeadLen~=(68+HDR.NS*96))
                        fprintf(HDR.FILE.stderr,'Error SOPEN CFWB: size of header2 does not fit in file %s\n',HDR.FileName);
                end;
        end;
        HDR.Calib = [HDR.Off';speye(HDR.NS)]*spdiags(1:HDR.NS,1:HDR.NS,HDR.Cal);
        HDR.Label = setstr(HDR.Label);
        HDR.PhysDim = setstr(HDR.PhysDim);
        
        HDR.HeadLen = ftell(HDR.FILE.FID);
        HDR.FILE.POS = 0; 
        HDR.AS.endpos = HDR.SPR; 
        
        
elseif strcmp(HDR.TYPE,'ISHNE'),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le')
                
                fprintf(HDR.FILE.stderr,'Format not tested yet. \nFor more information contact <a.schloegl@ieee.org> Subject: Biosig/Dataformats \n',PERMISSION);	
                
                HDR.FILE.OPEN = 1;
                fseek(HDR.FILE.FID,10,'bof');
                HDR.variable_length_block = fread(HDR.FILE.FID,1,'int32');		
                HDR.SPR = fread(HDR.FILE.FID,1,'int32');		
                HDR.NRec= 1;
                HDR.offset_variable_length_block = fread(HDR.FILE.FID,1,'int32');
                HDR.HeadLen = fread(HDR.FILE.FID,1,'int32');		
                HDR.VERSION = fread(HDR.FILE.FID,1,'int16');		
                %HDR.Patient.Name = fread(HDR.FILE.FID,80,'char');		
                %HDR.Surname = fread(HDR.FILE.FID,40,'char');		
                HDR.PID = fread(HDR.FILE.FID,20,'char');		
                HDR.Patient.Sex = fread(HDR.FILE.FID,1,'int16');		
                HDR.Patient.Race = fread(HDR.FILE.FID,1,'int16');		
                HDR.Patient.Birthday = fread(HDR.FILE.FID,3,'int16');		
                %HDR.Surname = fread(HDR.FILE.FID,40,'char')		
                Date = fread(HDR.FILE.FID,[1,3],'int16');		
                Date2 = fread(HDR.FILE.FID,[1,3],'int16');		
                Time = fread(HDR.FILE.FID,[1,3],'int16');		
                HDR.T0 = [Date([3,2,1]),Time];
                HDR.NS = fread(HDR.FILE.FID,1,'int16');		
                HDR.Lead.Specification = fread(HDR.FILE.FID,12,'int16');		
                HDR.Lead.Quality = fread(HDR.FILE.FID,12,'int16');		
                AmplitudeResolution = fread(HDR.FILE.FID,12,'int16');
                if any(HDR.Lead.AmplitudeResolution(HDR.NS+1:12)~=-9)
                        fprintf(HDR.FILE.stderr,'Warning: AmplitudeResolution and Number of Channels %i do not fit.\n',HDR.NS);
                        fclose(HDR.FILE.FID); 
                        HDR.FILE.FID = -1;	
                end;
                
                HDR.PacemakerCode = fread(HDR.FILE.FID,1,'int16');		
                HDR.TypeOfRecorder = fread(HDR.FILE.FID,40,'char');		
                HDR.SampleRate = fread(HDR.FILE.FID,1,'int16');		
                HDR.Proprietary_of_ECG = fread(HDR.FILE.FID,80,'char');		
                HDR.Copyright = fread(HDR.FILE.FID,80,'char');		
                HDR.reserved1 = fread(HDR.FILE.FID,80,'char');		
                if ftell(HDR.FILE.FID)~=HDR.offset_variable_length_block,
                        fprintf(HDR.FILE.stderr,'ERROR: length of fixed header does not fit %i %i \n',ftell(HDR.FILE.FID),HDR.offset_variable_length_block);
                        fclose(HDR.FILE.FID); 
                        HDR.FILE.FID = -1;	
                        return;
                end;
                HDR.VariableHeader=fread(HDR.FILE.FID,HDR.variable_length_block,'char');	
                if ftell(HDR.FILE.FID)~=HDR.HeadLen,
                        fprintf(HDR.FILE.stderr,'ERROR: length of variable header does not fit %i %i \n',ftell(HDR.FILE.FID),HDR.HeadLen);
                        fclose(HDR.FILE.FID); 
                        HDR.FILE.FID = -1;	
                        return;
                end;
                
                HDR.Cal = eye(AmplitudeResolution(HDR.InChanSelect))/1000;
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal,HDR.NS+1,HDR.NS);
                HDR.PhysDim = 'uV';
                HDR.AS.bpb = 2*HDR.NS;
                HDR.GDFTYP = 'int16';
                HDR.AS.endpos = 8+2+512+HDR.variable_length_block+HDR.NS*2*HDR.SPR;
                HDR.FLAG.TRIGGERED = 0;	% Trigger Flag
                
        else
                fprintf(HDR.FILE.stderr,'PERMISSION %s not supported\n',PERMISSION);	
        end;			
        
        
elseif strcmp(HDR.TYPE,'DDT'),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
                tmp = fread(HDR.FILE.FID,2,'int32');
                HDR.Version = tmp(1); 
                HDR.HeadLen = tmp(2); 
                HDR.SampleRate = fread(HDR.FILE.FID,1,'double');
                HDR.NS = fread(HDR.FILE.FID,1,'int32');
                HDR.T0 = fread(HDR.FILE.FID,[1,6],'int32');
                HDR.Gain = fread(HDR.FILE.FID,1,'int32');
                HDR.Comment = char(fread(HDR.FILE.FID,[1,128],'char'));
		tmp = fread(HDR.FILE.FID,[1,256],'uint8');
		if HDR.Version == 100, 
			HDR.Bits = 12; 
			HDR.Cal = 5/2048*HDR.Gain;
		elseif HDR.Version == 101, 
			HDR.Bits = tmp(1);
			HDR.Cal = 5*2^(1-HDR.Bits)/HDR.Gain;
		elseif HDR.Version == 102, 
			HDR.Bits = tmp(1);
			ChannelGain = tmp(2:65);
			HDR.Cal = 5000*2^(1-HDR.Bits)./(HDR.Gain*ChannelGain);
		elseif HDR.Version == 103, 
			HDR.Bits = tmp(1);
			ChannelGain = tmp(2:65);
			HDR.PhysMax = tmp(66:67)*[1;256]
			HDR.Cal = 5000*2^(1-HDR.Bits)./(HDR.Gain*ChannelGain);
		end;
		HDR.DigMax(1:HDR.NS) = 2^(HDR.Bits-1)-1;
		HDR.DigMin(1:HDR.NS) = -(2^(HDR.Bits-1));
		HDR.PhysMax = HDR.DigMax * HDR.Cal;
		HDR.PhysMin = HDR.DigMin * HDR.Cal;
		HDR.PhysDim = 'mV';
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal);

                HDR.AS.bpb = 2*HDR.NS; 
                HDR.GDFTYP = 3; 
                HDR.SPR = (HDR.FILE.size-HDR.HeadLen)/HDR.AS.bpb;
		HDR.NRec = 1; 
                HDR.AS.endpos = HDR.SPR; 
                status = fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
                HDR.FILE.POS = 0; 
                HDR.FILE.OPEN = 1; 
        end;			
        
        
elseif strcmp(HDR.TYPE,'NEX'),
        fprintf(HDR.FILE.stderr,'Warning: SOPEN (NEX) is still in testing phase.\n');	
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
                if HDR.FILE.FID<0,
                        return;
                end
                
                HDR.FILE.POS  = 0;
                HDR.NEX.magic = fread(HDR.FILE.FID,1,'int32');
                HDR.VERSION = fread(HDR.FILE.FID,1,'int32');
                HDR.NEX.comment = char(fread(HDR.FILE.FID,[1,256],'char'));
                HDR.NEX.SampleRate = fread(HDR.FILE.FID, 1, 'double');
                HDR.NEX.begintime = fread(HDR.FILE.FID, 1, 'int32');
                HDR.NEX.endtime = fread(HDR.FILE.FID, 1, 'int32');
                HDR.NEX.NS = fread(HDR.FILE.FID, 1, 'int32');
                status = fseek(HDR.FILE.FID, 260, 'cof');

                HDR.EVENT.DUR = [];
                HDR.EVENT.CHN = [];
                
                for k = 1:HDR.NEX.NS,
                        HDR.NEX.pos0(k) = ftell(HDR.FILE.FID);
                        HDR.NEX.type(k) = fread(HDR.FILE.FID, 1, 'int32');
                        HDR.NEX.version(k) = fread(HDR.FILE.FID, 1, 'int32');
                        Label(k,:) = fread(HDR.FILE.FID, [1 64], 'char');
                        HDR.NEX.offset(k)  = fread(HDR.FILE.FID, 1, 'int32');
                        HDR.NEX.nf(k)  = fread(HDR.FILE.FID, 1, 'int32');
                        reserved(k,:) = char(fread(HDR.FILE.FID, [1 32], 'char'));
                        HDR.NEX.SampleRate(k) = fread(HDR.FILE.FID, 1, 'double');
                        HDR.NEX.Cal(k) = fread(HDR.FILE.FID, 1, 'double');
                        HDR.NEX.SPR(k) = fread(HDR.FILE.FID, 1, 'int32');
                        HDR.NEX.h2(:,k)= fread(HDR.FILE.FID,19,'uint32');
                        %nm = fread(HDR.FILE.FID, 1, 'int32');
                        %nl = fread(HDR.FILE.FID, 1, 'int32');

                        HDR.NEX.pos(k) = ftell(HDR.FILE.FID);
%                        fseek(HDR.FILE.FID, HDR.NEX.pos0(k)+208,'bof');
                end;
                HDR.HeadLen = ftell(HDR.FILE.FID); 

                HDR.NEX.Label = char(Label);
                HDR.PhysDim   = 'mV';
                HDR.FILE.POS  = 0; 
                HDR.FILE.OPEN = 1; 
                HDR.NRec = 1;

                % select AD-channels only,
                CH = find(HDR.NEX.type==5);
                HDR.AS.chanreduce = cumsum(HDR.NEX.type==5);
                HDR.NS = length(CH); 
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.NEX.Cal(CH));
                HDR.Label = HDR.NEX.Label(CH,:);
                HDR.AS.SampleRate = HDR.NEX.SampleRate(CH);
                HDR.AS.SPR = HDR.NEX.SPR(CH); 
                HDR.SPR = 1;
		HDR.SampleRate = 1; 
                for k = 1:HDR.NS,
                        HDR.SPR = lcm(HDR.SPR,HDR.AS.SPR(k));
			HDR.SampleRate = lcm(HDR.SampleRate,HDR.AS.SampleRate(k));
                end;
        end;			
        
        
elseif strcmp(HDR.TYPE,'PLEXON'),
        if any(PERMISSION=='r'),
                fprintf(HDR.FILE.stderr,'Warning:  SOPEN (PLX) is still in testing phase.\n');	

                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
                if HDR.FILE.FID<0,
                        return;
                end
                H1 = fread(HDR.FILE.FID,2,'int32');
		HDR.magic = H1(1);
                HDR.Version = H1(2); 
                HDR.PLX.comment = fread(HDR.FILE.FID,128,'char');
                H1 = fread(HDR.FILE.FID,14,'int32');
                HDR.PLX.ADFrequency = H1(1); 
		HDR.PLX.NumDSPChannels = H1(2);
		HDR.PLX.NumEventChannels = H1(3);
		HDR.PLX.NumSlowChannels = H1(4);
		HDR.PLX.NumPointsWave = H1(5);
		HDR.PLX.NumPointsPreThr = H1(6);
                %HDR.NS = H1(2); 
                HDR.EVENT.N = H1(3);
                HDR.NS = H1(4); 
                HDR.PLX.wavlen = H1(5);
                HDR.TimeOffset = H1(6); 
                HDR.T0 = H1(7:12)';
                HDR.PLX.fastread = H1(13);
                HDR.PLX.WaveFormFreq = H1(14);
                HDR.PLX.LastTimeStamp = fread(HDR.FILE.FID,1,'double');
                H1 = fread(HDR.FILE.FID,4,'char');
	        H2 = fread(HDR.FILE.FID,3,'uint16');
		if HDR.Version>=103,
	                HDR.PLX.Trodalness          = H1(1);
    		        HDR.PLX.DataTrodalness      = H1(2);
        	        HDR.PLX.BitsPerSpikeSample  = H1(3);
        	        HDR.PLX.BitsPerSlowSample   = H1(4);
			HDR.PLX.SpikeMaxMagnitudeMV = H2(1);
			HDR.PLX.SlowMaxMagnitudeMV  = H2(2);
		end;	
		if HDR.Version>=105,
			HDR.PLX.SpikePreAmpGain     = H2(3);
		end;
                H1 = fread(HDR.FILE.FID,46,'char');
		
                HDR.PLX.tscount = fread(HDR.FILE.FID,[5,130],'int32');
                HDR.PLX.wfcount = fread(HDR.FILE.FID,[5,130],'int32');
                HDR.PLX.evcount = fread(HDR.FILE.FID,[1,300],'int32');
                HDR.PLX.adcount = fread(HDR.FILE.FID,[1,212],'int32');
        
                %HDR.PLX.dspHeader = fread(HDR.FILE.FID,[1020,HDR.NS],'uint8');
		for k = 1:HDR.PLX.NumDSPChannels,
            		tmp = fread(HDR.FILE.FID,[32,2],'char');
			HDR.Spike.Name(k,:) 	 = tmp(:,1)';
			HDR.Spike.SIGName(k,:) 	 = tmp(:,2)';
            		tmp = fread(HDR.FILE.FID,9,'int32');
			HDR.Spike.Channel(k) 	 = tmp(1);
			HDR.Spike.WFRate(k) 	 = tmp(2);
			HDR.Spike.SIG(k) 	 = tmp(3);
			HDR.Spike.Ref(k)         = tmp(4);
			HDR.Spike.Gain(k)        = tmp(5);
			HDR.Spike.Filter(k)      = tmp(6);
			HDR.Spike.Threshold(k)   = tmp(7);
			HDR.Spike.Method(k)      = tmp(8);
			HDR.Spike.NUnits(k)      = tmp(9);
            		HDR.Spike.template(k,:,:) = fread(HDR.FILE.FID,[5,64],'int16');
            		tmp = fread(HDR.FILE.FID,6,'int32');
			HDR.Spike.Fit(k,:)       = tmp(1:5)';
			HDR.Spike.SortWidth(k)   = tmp(6);
			HDR.Spike.Boxes(k,:,:,:) = reshape(fread(HDR.FILE.FID,[40],'int16'),[5,2,4]);
            		HDR.Spike.SortBeg(k)     = fread(HDR.FILE.FID,1,'int32');
            		HDR.Spike.Comment(k,:)   = fread(HDR.FILE.FID,[1,128],'char');
            		tmp = fread(HDR.FILE.FID,11,'int32');
		end;
		HDR.Spike.Name = deblank(char(HDR.Spike.Name));
		HDR.Spike.Comment = deblank(char(HDR.Spike.Comment));
		for k = 1:HDR.PLX.NumEventChannels,
			HDR.EV.Name(k,:)         = fread(HDR.FILE.FID,[1,32],'char');
			HDR.EV.Channel(k)        = fread(HDR.FILE.FID,1,'int32');
            		HDR.EV.Comment(k,:)      = fread(HDR.FILE.FID,[1,128],'char');
            		tmp = fread(HDR.FILE.FID,33,'int32');
		end;
		HDR.EV.Name = deblank(char(HDR.EV.Name));
		HDR.EV.Comment = deblank(char(HDR.EV.Comment));
		for k = 1:HDR.PLX.NumSlowChannels,
			HDR.Cont.Name(k,:) = fread(HDR.FILE.FID,[1,32],'char');
			tmp = fread(HDR.FILE.FID,6,'int32');
			HDR.Cont.Channel(k)      = tmp(1)+1;
			HDR.Cont.ADfreq(k)       = tmp(2);
			HDR.Cont.Gain(k)         = tmp(3);
			HDR.Cont.Enabled(k)      = tmp(4);
			HDR.Cont.PreAmpGain(k)   = tmp(5);
			HDR.Cont.SpikeChannel(k) = tmp(6);
            		HDR.Cont.Comment(k,:) 	 = fread(HDR.FILE.FID,[1,128],'char');
            		tmp = fread(HDR.FILE.FID,28,'int32');
		end;
		HDR.Cont.Name = deblank(char(HDR.Cont.Name));
		HDR.Cont.Comment = deblank(char(HDR.Cont.Comment));

                HDR.AS.SampleRate = HDR.Cont.ADfreq;
                HDR.HeadLen = ftell(HDR.FILE.FID); 
		HDR.EVENT.SampleRate = HDR.PLX.ADFrequency;
		HDR.PhysDim = 'mV';
		if HDR.Version<=102,
			HDR.Spike.Cal = 3./(2048*HDR.Spike.Gain);
		elseif HDR.Version<105
			HDR.Spike.Cal = HDR.PLX.SpikeMaxMagnitudeMV*2.^(-HDR.PLX.BitsPerSpikeSample)./(500*HDR.Spike.Gain);
		else
			HDR.Spike.Cal = HDR.PLX.SpikeMaxMagnitudeMV*2.^(1-HDR.PLX.BitsPerSpikeSample)./(HDR.Spike.Gain*HDR.PLX.SpikePreAmpGain);
		end;			
		if HDR.Version<=101,
			HDR.Cal = 5./(2048*HDR.Cont.Gain);
		elseif HDR.Version<=102,
			HDR.Cal = 5000./(2048*HDR.Cont.Gain.*HDR.Cont.PreAmpGain);
		else
			HDR.Cal = HDR.PLX.SpikeMaxMagnitudeMV*2^[1-HDR.PLX.BitsPerSlowSample]./(HDR.Cont.Gain.*HDR.Cont.PreAmpGain);
		end;			
                
                % transfrom into native format
                HDR.Label = HDR.Cont.Name;
                HDR.NRec = 1;
                HDR.SPR = max(HDR.PLX.adcount);
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal);
		HDR.SampleRate = 1;
		for k=1:HDR.NS,
			HDR.SampleRate = lcm(HDR.SampleRate,HDR.AS.SampleRate(k));
		end;
                HDR.FILE.POS = 0; 
                HDR.FILE.OPEN = 1; 

                CH = find(HDR.PLX.adcount>0);
                if isempty(ReRefMx) & any(CH) & (max(CH)<150),
                        HDR.NS = max(CH); 
                        HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
                        HDR.Label = HDR.Label(1:HDR.NS,:);
                end;
        end;			

        
elseif strcmp(HDR.TYPE,'Nicolet'),
        if any(PERMISSION=='r'),
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
        else
                fprintf(HDR.FILE.stderr,'PERMISSION %s not supported\n',PERMISSION);	
        end;			
        
        
elseif strncmp(HDR.TYPE,'SEG2',4),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb',HDR.Endianity);
                
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS  = 0;
                HDR.VERSION = fread(HDR.FILE.FID,1,'int16');
                HDR.HeadLen = fread(HDR.FILE.FID,1,'uint16');
                HDR.NS      = fread(HDR.FILE.FID,1,'uint16');
                HDR.SEG2.nsterm = fread(HDR.FILE.FID,1,'uint8'); 	% number of string terminator 
                HDR.SEG2.sterm  = fread(HDR.FILE.FID,2,'uchar'); 	% string terminator 
                HDR.SEG2.nlterm = fread(HDR.FILE.FID,1,'uint8'); 	% number of line terminator 
                HDR.SEG2.lterm  = fread(HDR.FILE.FID,2,'uchar'); 	% line terminator 
                HDR.SEG2.TraceDesc = fread(HDR.FILE.FID,HDR.NS,'uint32');
                
                % initialize date
                HDR.SEG2.blocksize = repmat(nan,HDR.NS,1);
                HDR.AS.bpb = repmat(nan,HDR.NS,1);
                HDR.AS.spb = repmat(nan,HDR.NS,1);
                HDR.SEG2.DateFormatCode = repmat(nan,HDR.NS,1);
                
                if ftell(HDR.FILE.FID) ~= HDR.HeadLen, 
                        fprintf(HDR.FILE.stderr,'Warning SOPEN TYPE=SEG2: headerlength does not fit.\n');
                end; 
                
                optstrings = fread(HDR.FILE.FID,HDR.SEG2.TraceDesc(1)-HDR.Headlen,'uchar');
                
                id_tmp = fread(HDR.FILE.FID,1,'uint16');
                if id_tmp ~=hex2dec('4422')
                        fprintf(HDR.FILE.stderr,'Error SOPEN TYPE=SEG2: incorrect trace descriptor block ID.\n');
                end;
                
                for k = 1:HDR.NS, 
                        fseek(HDR.FILE.FID,HDR.SEG2.TraceDesc(k),'bof');
                        HDR.SEG2.blocksize(k)  = fread(HDR.FILE.FID,1,'uint16');
                        HDR.AS.bpb(k)  = fread(HDR.FILE.FID,1,'uint32');
                        HDR.AS.spb(k)  = fread(HDR.FILE.FID,1,'uint32');
                        HDR.SEG2.DateFormatCode(k) = fread(HDR.FILE.FID,1,'uchar');
                        
                        fseek(HDR.FILE.FID,32-13,'cof');
                        %[tmp,c] = fread(HDR.FILE.FID,32-13,'char');	% reserved
                        
                        optstrings = fread(HDR.FILE.FID,HDR.SEG2.blocksize(k)-32,'uchar');
                end; 
                
                fprintf(HDR.FILE.stderr,'Format %s not implemented yet. \nFor more information contact <a.schloegl@ieee.org> Subject: Biosig/Dataformats \n',HDR.TYPE);	
                fclose(HDR.FILE.FID);
                HDR.FILE.FID = -1;
                HDR.FILE.OPEN = 0;
        end;		
        
        
elseif strncmp(HDR.TYPE,'SIGIF',5),
        if any(PERMISSION=='r'),
                HDR.FILE.FID  = fopen(HDR.FileName,'rb','ieee-le');
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS  = 0;
                
                HDR.fingerprint=fgetl(HDR.FILE.FID);   % 1
                
                if length(HDR.fingerprint)>6
                        HDR.VERSION = int2str(HDR.fingerprint(7));
                else
                        HDR.VERSION = 1.1;
                end;        
                HDR.Comment=fgetl(HDR.FILE.FID);		% 2        
                HDR.SignalName=fgetl(HDR.FILE.FID);	% 3
                HDR.Date=fgetl(HDR.FILE.FID);		% 4 
                HDR.modifDate=fgetl(HDR.FILE.FID);	% 5
                
                [tmp1,tmp] = strtok(HDR.Date,'-/'); 
                HDR.T0     = zeros(1,6);
                HDR.T0(1)  = str2double(tmp1);
                if length(tmp1)<3, HDR.T0(1) = 1900+HDR.T0(1); end;
                [tmp1,tmp] = strtok(tmp,'-/'); 
                HDR.T0(2)  = str2double(tmp1);
                [tmp1,tmp] = strtok(tmp,'-/'); 
                HDR.T0(3)  = str2double(tmp1);
                
                HDR.SIG.Type   = fgetl(HDR.FILE.FID);		% 6 simultaneous or serial sampling
                Source = fgetl(HDR.FILE.FID);		% 7 - obsolete
                HDR.NS     = str2double(fgetl(HDR.FILE.FID));  	% 8 number of channels
                HDR.NRec   = str2double(fgetl(HDR.FILE.FID)); % 9 number of segments
                NFrames= str2double(fgetl(HDR.FILE.FID));  % 10 number of frames per segment - obsolete
                
                %HDR.SPR    = str2double(fgetl(HDR.FILE.FID));  			% 11 	number of samples per frame
                HDR.AS.spb  = str2double(fgetl(HDR.FILE.FID));  			% 11 	number of samples per frame
                H1.Bytes_per_Sample = str2double(fgetl(HDR.FILE.FID));	% 12 number of bytes per samples
                HDR.AS.bpb = HDR.AS.spb * H1.Bytes_per_Sample;
                HDR.Sampling_order    = str2double(fgetl(HDR.FILE.FID));  	% 13
                HDR.FLAG.INTEL_format = str2double(fgetl(HDR.FILE.FID));  	% 14
                HDR.FormatCode = str2double(fgetl(HDR.FILE.FID));  	% 15
                
                HDR.CompressTechnique = fgetl(HDR.FILE.FID);  		% 16
                HDR.SignalType = fgetl(HDR.FILE.FID);  			% 17
                
                for k=1:HDR.NS,
                        chandata = fgetl(HDR.FILE.FID);			% 18
                        [tmp,chandata] = strtok(chandata,' ,;:');  
                        HDR.Label(k,1:length(tmp)) = tmp;
                        [tmp,chandata] = strtok(chandata,' ,;:');  
                        HDR.Cal(k) = str2double(tmp);  
                        
                        [tmp,chandata] = strtok(chandata,' ,;:');
                        HDR.SampleRate(k) = str2double(tmp);
                        
                        %[tmp,chandata] = strtok(chandata);  
                        HDR.Variable{k} = chandata;  
                        
                        while  ~isempty(chandata)
                                [tmp,chandata] = strtok(chandata,' ,;:'); 
                                if strcmp(tmp,'G')
                                        [HDR.PhysDim{k},chandata] = strtok(chandata,' ,;:');  
                                end;        
                        end;
                end;
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal,HDR.NS+1,HDR.NS);
                HDR.Segment_separator = fgetl(HDR.FILE.FID);  		% 19
                %HDR.Segment_separator = hex2dec(fgetl(HDR.FILE.FID));  
                
                HDR.FLAG.TimeStamp = str2double(fgetl(HDR.FILE.FID));  	% 20
                
                if HDR.VERSION>=3,
                        HDR.FLAG.SegmentLength = str2double(fgetl(HDR.FILE.FID));	% 21  
                        HDR.AppStartMark = fgetl(HDR.FILE.FID);  		% 22
                        HDR.AppInfo = fgetl(HDR.FILE.FID);  			% 23
                else
                        HDR.FLAG.SegmentLength = 0;    
                end;        
                HDR.footer = fgets(HDR.FILE.FID,6);			% 24
                
                if ~strcmp(HDR.footer,'oFSvAI')
                        fprintf(HDR.FILE.stderr,'Warning LOADSIG in %s: Footer not found\n',  HDR.FileName);  
                end;
                
                if HDR.VERSION<2,
                        HDR.FLAG.SegmentLength = 0;
                end;
                
                switch HDR.FormatCode,
                        case 0; HDR.GDFTYP = 'uint16';
                        case 3; HDR.GDFTYP = 'int16';  
                                HDR.Segment_separator = hex2dec(HDR.Segment_separator([3:4,1:2]));
                        case 5; HDR.GDFTYP = 'float';
                        otherwise;
                                fprintf(HDR.FILE.stderr,'Warning LOADSIG: FormatCode %i not implemented\n',HDR.FormatCode);
                end;
                
                tmp = ftell(HDR.FILE.FID);
                if ~HDR.FLAG.INTEL_format,
                        fclose(HDR.FILE.FID);
                        HDR.FILE.FID = fopen(HDR.FileName,'rt','ieee-be');
                        fseek(HDR.FILE.FID,tmp,'bof');
                end;
                HDR.HeadLen = tmp + HDR.FLAG.TimeStamp*9;
                
                if ~HDR.NRec, HDR.NRec = inf; end;
                k = 0;
                while (k < HDR.NRec) & ~feof(HDR.FILE.FID),
                        k = k+1;
                        HDR.Block.Pos(k) = ftell(HDR.FILE.FID);
                        if HDR.FLAG.TimeStamp,
                                HDR.Frame(k).TimeStamp = fread(HDR.FILE.FID,[1,9],'char');
                        end;
                        
                        if HDR.FLAG.SegmentLength,
                                HDR.Block.Length(k) = fread(HDR.FILE.FID,1,'uint16');  %#26
                                fseek(HDR.FILE.FID,HDR.Block.Length(k)*H1.Bytes_per_Sample,'cof');
                        else
                                tmp = HDR.Segment_separator-1;
                                count = 0;
                                data  = [];
                                dat   = [];
                                while ~(any(dat==HDR.Segment_separator));
                                        [dat,c] = fread(HDR.FILE.FID,[HDR.NS,1024],HDR.GDFTYP);
                                        count   = count + c;
                                end;
                                tmppos = min(find(dat(:)==HDR.Segment_separator));
                                HDR.Block.Length(k) = count - c + tmppos;
                        end;
                end;
                HDR.SPR = HDR.Block.Length/HDR.NS;
                HDR.Dur = max(HDR.SPR./HDR.SampleRate);
                HDR.NRec = k; 
                
                if HDR.FLAG.TimeStamp,
                        tmp=char(HDR.Frame(1).TimeStamp);
                        HDR.T0(4) = str2double(tmp(1:2));
                        HDR.T0(5) = str2double(tmp(3:4));
                        HDR.T0(6) = str2double([tmp(5:6),'.',tmp(7:9)]);
                end;
        end;
        
elseif strcmp(HDR.TYPE,'CTF'),
        if any(PERMISSION=='r'),
                HDR.FILE.FID  = fopen(fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.res4']),'rb','ieee-be');
		if HDR.FILE.FID<0,
			return
		end;
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS  = 0;
		fseek(HDR.FILE.FID,778,'bof');
		tmp = char(fread(HDR.FILE.FID,255,'char')');
                tmp(tmp==':')=' ';
                tmp = str2double(tmp);
                if length(tmp)==3,
                        HDR.T0(4:6)=tmp;
                end;
		tmp = char(fread(HDR.FILE.FID,255,'char')');
                tmp(tmp=='/')=' ';
                tmp = str2double(tmp);
                if length(tmp)==3,
                        HDR.T0(1:3) = tmp;
                end;
                
		HDR.SPR = fread(HDR.FILE.FID,1,'int32');
		HDR.NS = fread(HDR.FILE.FID,1,'int16');
		HDR.CTF.NS2 = fread(HDR.FILE.FID,1,'int16');
		HDR.SampleRate = fread(HDR.FILE.FID,1,'double');
		HDR.Dur = fread(HDR.FILE.FID,1,'double');
		HDR.NRec = fread(HDR.FILE.FID,1,'int16');
		HDR.CTF.NRec2 = fread(HDR.FILE.FID,1,'int16');
		HDR.TriggerOffset = fread(HDR.FILE.FID,1,'int32');
		
		fseek(HDR.FILE.FID,1712,'bof');
		HDR.PID = char(fread(HDR.FILE.FID,32,'char')');
		HDR.Operator = char(fread(HDR.FILE.FID,32,'char')');
		HDR.FILE.SensorFileName = char(fread(HDR.FILE.FID,60,'char')');

		%fseek(HDR.FILE.FID,1836,'bof');
		HDR.CTF.RunSize = fread(HDR.FILE.FID,1,'int32');
		HDR.CTF.RunSize2 = fread(HDR.FILE.FID,1,'int32');
		HDR.CTF.RunDescription = char(fread(HDR.FILE.FID,HDR.CTF.RunSize,'char')');
		HDR.CTF.NumberOfFilters = fread(HDR.FILE.FID,1,'int16');
		
		for k = 1:HDR.CTF.NumberOfFilters,
			F.Freq = fread(HDR.FILE.FID,1,'double');
			F.Class = fread(HDR.FILE.FID,1,'int32');
			F.Type = fread(HDR.FILE.FID,1,'int32');
			F.NA = fread(HDR.FILE.FID,1,'int16');
			F.A = fread(HDR.FILE.FID,[1,F.NA],'double');
			HDR.CTF.Filter(k) = F; 
		end;
		
		tmp = fread(HDR.FILE.FID,[32,HDR.NS],'char');
		tmp(tmp<0) = 0;
		tmp(tmp>127) = 0;
		tmp(cumsum(tmp==0)>0)=0;
		HDR.Label = char(tmp');
		
		for k = 1:HDR.NS,
			info.index(k,:) = fread(HDR.FILE.FID,1,'int16');
			info.extra(k,:) = fread(HDR.FILE.FID,1,'int16');
			info.ix(k,:) = fread(HDR.FILE.FID,1,'int32');
			info.gain(k,:) = fread(HDR.FILE.FID,[1,4],'double');

			info.index2(k,:) = fread(HDR.FILE.FID,1,'int16');
			info.extra2(k,:) = fread(HDR.FILE.FID,1,'int16');
			info.ix2(k,:) = fread(HDR.FILE.FID,1,'int32');

			fseek(HDR.FILE.FID,1280,'cof');
		end;
		fclose(HDR.FILE.FID);
                
                %%%%% read Markerfile %%%%%
                fid = fopen(fullfile(HDR.FILE.Path,'MarkerFile.mrk'),'rb','ieee-be');
                if fid > 0,
                        while ~feof(fid),
                                s = fgetl(fid);
                                if ~isempty(strmatch('PATH OF DATASET:',s))
                                        file = fgetl(fid);
                                        
                                elseif 0, 
                                        
                                elseif ~isempty(strmatch('TRIAL NUMBER',s))
                                        N = 0; 
                                        x = fgetl(fid);
                                        while ~isempty(x),
                                                tmp = str2double(x);
                                                N = N+1;
                                                HDR.EVENT.POS(N,1) = tmp(1)*HDR.SPR+tmp(2)*HDR.SampleRate;
                                                HDR.EVENT.TYP(N,1) = 1;
                                                x = fgetl(fid);
                                        end
                                else
                                        
                                end
                        end
                        fclose(fid);
                end;
                
		HDR.CTF.info = info;
		ix = (info.index==0) | (info.index==1) | (info.index==9);
		ix0 = find(ix);
		HDR.Cal(ix0) = 1./(info.gain(ix0,1) .* info.gain(ix0,2));
		ix0 = find(~ix);
		HDR.Cal(ix0) = 1./info.gain(ix0,2);
		HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal);
		HDR.FLAG.TRIGGERED = HDR.NRec > 1;
		HDR.AS.spb = HDR.NRec * HDR.NS;
		HDR.AS.bpb = HDR.AS.spb * 4; 
		
		HDR.CHANTYP = char(repmat(32,HDR.NS,1));
		HDR.CHANTYP(info.index==9) = 'E';
		HDR.CHANTYP(info.index==5) = 'M';
		HDR.CHANTYP(info.index==1) = 'R';
		HDR.CHANTYP(info.index==0) = 'R';

		if 0,

		elseif strcmpi(CHAN,'MEG'),
			CHAN = find(info.index==5); 
		elseif strcmpi(CHAN,'EEG'),
			CHAN = find(info.index==9); 
		elseif strcmpi(CHAN,'REF'),
			CHAN = find((info.index==0) | (info.index==1)); 
		elseif strcmpi(CHAN,'other'),
			CHAN = find((info.index~=0) & (info.index~=1) & (info.index~=5) & (info.index~=9)); 
		end;	
		
                HDR.FILE.FID = fopen(fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.meg4']),'rb','ieee-be');
		HDR.VERSION = char(fread(HDR.FILE.FID,[1,8],'char'));
		HDR.HeadLen = ftell(HDR.FILE.FID);
		fseek(HDR.FILE.FID,0,'eof');
		HDR.AS.endpos = ftell(HDR.FILE.FID);
		fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
        end;

        
elseif strcmp(HDR.TYPE,'BrainVision'),
        % get the header information from the VHDR ascii file
        fid = fopen(HDR.FileName,'rt');
        if fid<0,
                fprintf('Error SOPEN: could not open file %s\n',HDR.FileName);
                return;
        end; 
        tline = fgetl(fid);
        HDR.BV = [];
        UCAL = 0; 
        flag = 1; 
        while ~feof(fid), 
                tline = fgetl(fid);
                if isempty(tline),
                elseif tline(1)==';',
                elseif tline(1)==10, 
                elseif tline(1)==13,    % needed for Octave 
                elseif strncmp(tline,'[Common Infos]',14)
                        flag = 2;     
                elseif strncmp(tline,'[Binary Infos]',14)
                        flag = 3;     
                elseif strncmp(tline,'[Channel Infos]',14)
                        flag = 4;     
                elseif strncmp(tline,'[Coordinates]',12)
                        flag = 5;     
                elseif strncmp(tline,'[Marker Infos]',12)
                        flag = 6;     
                elseif strncmp(tline,'[Comment]',9)
                        flag = 7;     
                elseif strncmp(tline,'[',1)     % any other segment
                        flag = 8;     
                        
                elseif any(flag==[2,3]),
                        [t1,r] = strtok(tline,'=');
                        [t2,r] = strtok(r,['=,',char([10,13])]);
                        if ~isempty(t2),
                                HDR.BV = setfield(HDR.BV,t1,t2);
                        end;
                elseif flag==4,        
                        [t1,r] = strtok(tline,'=');
                        [t2,r] = strtok(r, ['=',char([10,13])]);
                        ix = [find(t2==','),length(t2)];
                        [chan, stat1] = str2double(t1(3:end));
                        HDR.Label{chan,1} = [t2(1:ix(1)-1),' '];
                        HDR.BV.reference{chan,1} = t2(ix(1)+1:ix(2)-1);
                        [v, stat] = str2double(t2(ix(2)+1:end));          % in microvolt
                        if (prod(size(v))==1) & ~any(stat)
                                HDR.Cal(chan) = v;                                
                        else
                                UCAL = 1; 
                                HDR.Cal(chan) = 1;
                        end;
                elseif flag==5,   
                        [t1,r] = strtok(tline,'=');
                        chan = str2double(t1(3:end));
                        [v, stat] = str2double(r(2:end));
                        HDR.ELPOS(chan,:) = v;
                end
        end
        fclose(fid);

        % convert the header information to BIOSIG standards
        HDR.NS = str2double(HDR.BV.NumberOfChannels);
        HDR.SampleRate = 1e6/str2double(HDR.BV.SamplingInterval);      % sampling rate in Hz
        if UCAL & ~strncmp(HDR.BV.BinaryFormat,'IEEE_FLOAT',10),
                fprintf(2,'Warning SOPEN (BV): missing calibration values\n');
                HDR.FLAG.UCAL = 1; 
        end;
        HDR.NRec = 1;                   % it is a continuous datafile, therefore one record
        HDR.Calib = [zeros(1,HDR.NS) ; diag(HDR.Cal)];  % is this correct?
        HDR.PhysDim = 'uV';
        HDR.FLAG.TRIGGERED = 0; 
        HDR.Filter.LowPass = repmat(NaN,HDR.NS,1);
        HDR.Filter.HighPass = repmat(NaN,HDR.NS,1);
        HDR.Filter.Notch = repmat(NaN,HDR.NS,1);
        
        if strncmpi(HDR.BV.BinaryFormat, 'int_16',6)
                HDR.GDFTYP = 'int16'; 
                HDR.AS.bpb = HDR.NS * 2; 
        elseif strncmpi(HDR.BV.BinaryFormat, 'ieee_float_32',13)
                HDR.GDFTYP = 'float32'; 
                HDR.AS.bpb = HDR.NS * 4; 
        elseif strncmpi(HDR.BV.BinaryFormat, 'ieee_float_64',13)
                HDR.GDFTYP = 'float64'; 
                HDR.AS.bpb = HDR.NS * 8; 
        end
        
        %read event file 
        fid = fopen(fullfile(HDR.FILE.Path, HDR.BV.MarkerFile),'rt');
        if fid>0,
                while ~feof(fid),
                        s = fgetl(fid);
                        if strncmp(s,'Mk',2),
                                [N,s] = strtok(s(3:end),'=');
                                ix = find(s==',');
                                ix(length(ix)+1)=length(s)+1;
                                N = str2double(N);
                                HDR.EVENT.POS(N,1) = str2double(s(ix(2)+1:ix(3)-1));
                                HDR.EVENT.TYP(N,1) = 0;
                                HDR.EVENT.DUR(N,1) = str2double(s(ix(3)+1:ix(4)-1));
                                HDR.EVENT.CHN(N,1) = str2double(s(ix(4)+1:ix(5)-1));
                                HDR.EVENT.TeegType{N,1} = s(2:ix(1)-1);
                                HDR.EVENT.TeegDesc{N,1} = s(ix(1)+1:ix(2)-1);
                        end;
                end
                fclose(fid);
        end

        %open data file 
        if strncmpi(HDR.BV.DataFormat, 'binary',5)
                PERMISSION='rb';
        elseif strncmpi(HDR.BV.DataFormat, 'ascii',5)                 
                PERMISSION='rt';
        end;

        HDR.FILE.FID = fopen(fullfile(HDR.FILE.Path,HDR.BV.DataFile),PERMISSION,'ieee-le');
        if HDR.FILE.FID < 0,
                fprintf(HDR.FILE.stderr,'ERROR SOPEN BV: could not open file %s\n',fullfile(HDR.FILE.Path,HDR.BV.DataFile));
                return;
        end;
        
        HDR.FILE.OPEN= 1; 
        HDR.FILE.POS = 0; 
        HDR.HeadLen  = 0; 
        if strncmpi(HDR.BV.DataFormat, 'binary',5)
                fseek(HDR.FILE.FID,0,'eof');
                HDR.AS.endpos = ftell(HDR.FILE.FID);
                fseek(HDR.FILE.FID,0,'bof');
                HDR.AS.endpos = HDR.AS.endpos/HDR.AS.bpb;
                
        elseif strncmpi(HDR.BV.DataFormat, 'ascii',5)  
                s = char(sread(HDR.FILE.FID,inf,'char')');
                s(s==',')='.';
                [tmp,status] = str2double(s);
                if strncmpi(HDR.BV.DataOrientation, 'multiplexed',6),
                        HDR.BV.data = tmp;
                elseif strncmpi(HDR.BV.DataOrientation, 'vectorized',6),
                        HDR.BV.data = tmp';
                end
                HDR.AS.endpos = size(HDR.BV.data,1);
                if ~any(HDR.NS ~= size(tmp));
                        fprintf(HDR.FILE.stderr,'ERROR SOPEN BV-ascii: number of channels inconsistency\n');
                end;
        end
        HDR.SPR = HDR.AS.endpos;
        
        
elseif strcmp(HDR.TYPE,'EEProbe-CNT'),

        if 1, %try
                % Read the first sample of the file with a mex function
                % this also gives back header information, which is needed here
                tmp = read_eep_cnt(HDR.FileName, 1, 1);

                % convert the header information to BIOSIG standards
                HDR.FILE.FID = 1;               % ?
                HDR.FILE.POS = 0;
                HDR.NS = tmp.nchan;             % number of channels
                HDR.SampleRate = tmp.rate;      % sampling rate
                HDR.NRec = 1;                   % it is always continuous data, therefore one record
                HDR.FLAG.TRIGGERED = 0; 
                HDR.SPR = tmp.nsample;          % total number of samples in the file
                HDR.Dur = tmp.nsample/tmp.rate; % total duration in seconds
                HDR.Calib = [zeros(1,HDR.NS) ; eye(HDR.NS, HDR.NS)];  % is this correct?
                HDR.Label = char(tmp.label);
                HDR.PhysDim = 'uV';
                HDR.AS.endpos = HDR.SPR;
                HDR.Label = tmp.label;
                
        else %catch
                fprintf(HDR.FILE.stderr,'Warning SOPEN (EEProbe): only experimental version. \n');

                HDR.FILE.FID = fopen(HDR.FileName,'rb');
                H = openiff(HDR.FILE.FID);
                if isfield(H,'RIFF');
                        HDR.FILE.OPEN = 1; 
                        HDR.RIFF = H.RIFF;
                        HDR.Label = {};
                        HDR.PhysDim = {};
                        if isfield(HDR.RIFF,'CNT');
                                if isfield(HDR.RIFF.CNT,'eeph');
				if ~isstruct(HDR.RIFF.CNT.eeph);
                                        rest = char(HDR.RIFF.CNT.eeph');                        
                                        while ~isempty(rest), 
                                                [tline,rest] = strtok(rest,[10,13]);
                                                if isempty(tline),
                                                        
                                                elseif strncmp(tline,'[Sampling Rate]',13)
                                                        [tline,rest] = strtok(rest,[10,13]);
                                                        [HDR.SampleRate,status] = str2double(tline);
                                                elseif strncmp(tline,'[Samples]',7)
                                                        [tline,rest] = strtok(rest,[10,13]);
                                                        [HDR.SPR,status] = str2double(tline);
                                                elseif strncmp(tline,'[Channels]',8)
                                                        [tline,rest] = strtok(rest,[10,13]);
                                                        [HDR.NS,status] = str2double(tline);
                                                elseif strncmp(tline,'[Basic Channel Data]',16)
                                                        while rest(2)==';'
                                                                [tline,rest] = strtok(rest,[10,13]);
                                                        end;
                                                        k = 1; 
                                                        while k<=HDR.NS,
                                                                [tline,rest] = strtok(rest,[10,13]);
                                                                [HDR.Label{k,1}, R] = strtok(tline,[9,10,13,32]);    
                                                                [Dig , R]  = strtok(R,[9,10,13,32]);    
                                                                [Phys, R]  = strtok(R,[9,10,13,32]);    
                                                                HDR.Cal(k) = str2double(Phys)/str2double(Dig); 
                                                                [HDR.PhysDim{k,1}, R] = strtok(R,[9,10,13,32]);    
                                                                k = k + 1;
                                                        end;
                                                        HDR.Calib = [zeros(1,HDR.NS);diag(HDR.Cal)];
                                                elseif strncmp(tline,'[History]',9)
                                                else
                                                end
                                        end;
                                end
				end;
                        end
                end
        end

        fid = fopen(fullfile(HDR.FILE.Path,[HDR.FILE.Name,'.trg']),'rt');
        if fid>0,
                header = fgetl(fid);
                N = 0; 
                while ~feof(fid),
                        tmp = fscanf(fid, '%f %d %s', 3);
                        if ~isempty(tmp)
                                N = N + 1; 
                                HDR.EVENT.POS(N,1) = tmp(1)*HDR.SampleRate;
                                HDR.EVENT.TYP(N,1) = 0;
                                %HDR.EVENT.DUR(N,1) = 0;
                                %HDR.EVENT.CHN(N,1) = 0;
                                
                                HDR.EVENT.TeegType{N,1}   = char(tmp(3:end));		% string
                                HDR.EVENT.TYP(N,1) = str2double(HDR.EVENT.TeegType{N,1});		% numeric
                        end
                end
                fclose(fid);
        end;
        
                
elseif strcmp(HDR.TYPE,'EEProbe-AVR'),
        % it appears to be a EEProbe file with an averaged ERP
        try
                tmp = read_eep_avr(HDR.FileName);
        catch
                fprintf(HDR.FILE.stderr,'ERROR SOPEN (EEProbe): Cannot open EEProbe-file, because read_eep_avr.mex not installed. \n');
                fprintf(HDR.FILE.stderr,'ERROR SOPEN (EEProbe): see http://www.smi.auc.dk/~roberto/eeprobe/\n');
                return;
        end

        % convert the header information to BIOSIG standards
        HDR.FILE.FID = 1;               % ?
        HDR.FILE.POS = 0;
        HDR.NS = tmp.nchan;             % number of channels
        HDR.SampleRate = tmp.rate;      % sampling rate
        HDR.NRec  = 1;                   % it is an averaged ERP, therefore one record
        HDR.SPR   = tmp.npnt;             % total number of samples in the file
        HDR.Dur   = tmp.npnt/tmp.rate;    % total duration in seconds
        HDR.Calib = [zeros(1,HDR.NS) ; eye(HDR.NS, HDR.NS)];  % is this correct?
        HDR.Label = char(tmp.label);
        HDR.PhysDim   = 'uV';
        HDR.FLAG.UCAL = 1;
        HDR.FILE.POS  = 0; 
        HDR.AS.endpos = HDR.SPR;
        HDR.Label = tmp.label;
        HDR.TriggerOffset = 0; 
        
        HDR.EEP.data = tmp.data';
        
        
elseif strncmp(HDR.TYPE,'FIF',3),
        if any(exist('rawdata')==[3,6]),
		if isempty(FLAG_NUMBER_OF_OPEN_FIF_FILES)
			FLAG_NUMBER_OF_OPEN_FIF_FILES = 0;
		end;	    
                if ~any(FLAG_NUMBER_OF_OPEN_FIF_FILES==[0,1])
                        fprintf(HDR.FILE.stderr,'ERROR SOPEN (FIF): number of open FIF files must be zero or one\n\t Perhaps, you forgot to SCLOSE(HDR) the previous FIF-file.\n');
                        return;
                end;

                try
                        rawdata('any',HDR.FileName);  % opens file 
                catch
                        tmp = which('rawdata');
                        [p,f,e]=fileparts(tmp);
                        fprintf(HDR.FILE.stderr,'ERROR SOPEN (FIF): Maybe you forgot to do \"export LD_LIBRARY_PATH=%s/i386 \" before you started Matlab. \n',p);
                        return
                end
                HDR.FILE.FID = 1;
                HDR.SampleRate = rawdata('sf');
                HDR.AS.endpos = rawdata('samples');
                [HDR.MinMax,HDR.Cal] = rawdata('range');
                [HDR.Label, type, number] = channames(HDR.FileName);
        
                rawdata('goto', 0);
                [buf, status] = rawdata('next'); 
                HDR.Dur = rawdata('t');
                [HDR.NS,HDR.SPR] = size(buf);
                HDR.AS.bpb = HDR.NS * 2;
                HDR.Calib = [zeros(1,HDR.NS);diag(HDR.Cal)]; 
                
                rawdata('goto', 0);
                HDR.FILE.POS = 0; 
                HDR.FILE.OPEN = 1; 
                FLAG_NUMBER_OF_OPEN_FIF_FILES = FLAG_NUMBER_OF_OPEN_FIF_FILES+1; 
                
        else
                fprintf(HDR.FILE.stderr,'ERROR SOPEN (FIF): NeuroMag FIFF access functions not available. \n');
                fprintf(HDR.FILE.stderr,'\tOnline available at: http://www.kolumbus.fi/kuutela/programs/meg-pd/ \n');
                return;
        end
        
        
elseif strncmp(HDR.TYPE,'FS3',3),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-be');
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS  = 0;
                HDR.Date = fgets(HDR.FILE.FID);
                HDR.Info = fgets(HDR.FILE.FID);
                HDR.SURF.N = fread(HDR.FILE.FID,1,'int32');
                HDR.FACE.N = fread(HDR.FILE.FID,1,'int32');
                HDR.VERTEX.COORD =   fread(HDR.FILE.FID,3*HDR.SURF.N,'float32');
                
                HDR.FACES = fread(HDR.FILE.FID,[3,HDR.FACE.N],'int32')';
                fclose(HDR.FILE.FID);
        end
        
        
elseif strncmp(HDR.TYPE,'FS4',3),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-be');
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS  = 0;
                
                tmp = fread(HDR.FILE.FID,[1,3],'uint8');
                HDR.SURF.N = tmp*(2.^[16;8;1]);
                tmp = fread(HDR.FILE.FID,[1,3],'uint8');
                HDR.FACE.N = tmp*(2.^[16;8;1]);
                HDR.VERTEX.COORD = fread(HDR.FILE.FID,3*HDR.SURF.N,'int16')./100;
                tmp = fread(HDR.FILE.FID,[4*HDR.FACE.N,3],'uint8')*(2.^[16;8;1]);
                HDR.FACES = reshape(tmp,4,HDR.FACE.N)';
                fclose(HDR.FILE.FID);
        end
        
        
elseif strncmp(HDR.TYPE,'TRI',3),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rb','ieee-le');
                HDR.FILE.POS  = 0;
                
                HDR.ID = fread(HDR.FILE.FID,1,'int32');
                HDR.type = fread(HDR.FILE.FID,1,'short');
                HDR.VERSION = fread(HDR.FILE.FID,1,'short');
                HDR.ELEC.Thickness = fread(HDR.FILE.FID,1,'float');
                HDR.ELEC.Diameter = fread(HDR.FILE.FID,1,'float');
                HDR.reserved = fread(HDR.FILE.FID,4080,'char');
                
                HDR.FACE.N = fread(HDR.FILE.FID,1,'short');
                HDR.SURF.N = fread(HDR.FILE.FID,1,'short');
                
                HDR.centroid = fread(HDR.FILE.FID,[4,HDR.FACE.N],'float')';
                HDR.VERTICES = fread(HDR.FILE.FID,[4,HDR.SURF.N],'float')';
                HDR.FACES = fread(HDR.FILE.FID,[3,HDR.FACE.N],'short')';
                
                HDR.ELEC.N = fread(HDR.FILE.FID,1,'ushort');
                for k = 1:HDR.ELEC.N,
                        tmp = fread(HDR.FILE.FID,[1,10],'uchar');
                        Label{k,1} = [strtok(tmp,0), ' '];
                        HDR.ELEC.Key(k,1)  = fread(HDR.FILE.FID,1,'short');	
                        tmp = fread(HDR.FILE.FID,[1,3],'float');	
                        % HDR.elec(k).POS  = tmp(:);	
                        HDR.ELEC.XYZ(k,:)  = tmp;
                        HDR.ELEC.CHAN(k,1) = fread(HDR.FILE.FID,1,'ushort');	
                end;
                fclose(HDR.FILE.FID);
                HDR.Label = strvcat(Label);
                HDR.TYPE = 'ELPOS'; 
        end
        
        
elseif strcmp(HDR.TYPE,'DICOM'),
	HDR = opendicom(HDR,PERMISSION);
        
        
elseif 0, strcmp(HDR.TYPE,'DXF'),
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rt','ieee-le');
                
		while ~feof(HDR.FILE.FID),
	                line1 = fgetl(HDR.FILE.FID);
    		        line2 = fgetl(HDR.FILE.FID);
			
			[val,status] = str2double(line1);
			
			if any(status),
				error('SOPEN (DXF)');
			elseif val==999, 
			
			elseif val==0, 
			
			elseif val==1, 
			
			elseif val==2, 
			
			else
			
			end;
		end;
		
                fclose(HDR.FILE.FID);
        end


elseif strcmp(HDR.TYPE,'STX'),
        if any(PERMISSION=='r'),
                fid = fopen(HDR.FileName,'rt','ieee-le');
                FileInfo = fread(fid,20,'char');
                HDR.Label = fread(fid,50,'char');
                tmp = fread(fid,6,'int');
                HDR.NRec = tmp(1);
		HDR.SPR = 1; 
		
		tmp = fread(fid,5,'long');
		HDR.HeadLen = 116;

		
                fclose(HDR.FILE.FID);
        end


elseif strcmp(HDR.TYPE,'ABF'), 
        fprintf(HDR.FILE.stderr,'Warning: SOPEN (ABF) is still experimental.\n');	
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rt','ieee-le');
                HDR.ABF.ID = fread(HDR.FILE.FID,1,'uint32');
                HDR.Version = fread(HDR.FILE.FID,1,'float32');
                HDR.ABF.Mode = fread(HDR.FILE.FID,1,'uint16');
                HDR.AS.endpos = fread(HDR.FILE.FID,1,'uint32');
                HDR.ABF.NumPoinstsIgnored = fread(HDR.FILE.FID,1,'uint16');
                HDR.NRec = fread(HDR.FILE.FID,1,'uint32');
                t = fread(HDR.FILE.FID,3,'uint32');
                HDR.T0(1:3) = [floor(t(1)/1e4), floor(mod(t(1),1e4)/100), mod(t(1),100)];
                HDR.T0(4:6) = [floor(t(2)/3600),floor(mod(t(2),3600)/60),mod(t(2),60)];
                if HDR.T0(1)<80, HDR.T0(1)=HDR.T0(1)+2000;
                elseif HDR.T0(1)<100, HDR.T0(1)=HDR.T0(1)+1900;
                end;
                
                HDR.ABF.HeaderVersion = fread(HDR.FILE.FID,1,'float32');
		if HDR.ABF.HeaderVersion>=1.6,
			HDR.HeadLen = 1394+6144+654;
		else
			HDR.HeadLen =  370+2048+654;
		end;	
                HDR.ABF.FileType = fread(HDR.FILE.FID,1,'uint16');
                HDR.ABF.MSBinFormat = fread(HDR.FILE.FID,1,'uint16');

                HDR.ABF.SectionStart = fread(HDR.FILE.FID,15,'uint32');
                DataFormat = fread(HDR.FILE.FID,1,'uint16');
                HDR.ABF.simultanousScan = fread(HDR.FILE.FID,1,'uint16');
                t = fread(HDR.FILE.FID,4,'uint32');

                HDR.NS = fread(HDR.FILE.FID,1,'uint16');		
                tmp = fread(HDR.FILE.FID,4,'float32');

		HDR.SampleRate = 1000/tmp(1);
		if ~DataFormat
			HDR.GDFTYP = 3;		% int16 
			HDR.AS.bpb = 2*HDR.NS;
		else
			HDR.GDFTYP = 16;	% float32
			HDR.AS.bpb = 4*HDR.NS;
		end;	
		HDR.SPR = HDR.AS.endpos/HDR.NRec; 
		HDR.Dur = HDR.SPR/HDR.SampleRate;
		if HDR.FILE.size ~= HDR.HeadLen + HDR.AS.bpb*HDR.NRec*HDR.SPR;
			[HDR.FILE.size,HDR.HeadLen,HDR.AS.bpb*HDR.NRec*HDR.SPR]
			fprintf(HDR.FILE.stderr,'Warning SOPEN (ABF): filesize does not fit.\n');
		end;

                t = fread(HDR.FILE.FID,5,'uint32');

                t = fread(HDR.FILE.FID,3,'uint16');
		HDR.FLAG.AVERAGE = t(1);
		
                HDR.TRIGGER.THRESHOLD = fread(HDR.FILE.FID,1,'float');
                t = fread(HDR.FILE.FID,3,'uint16');

% this part is from IMPORT_ABF.M from 
%     © 2002 - Michele Giugliano, PhD (http://www.giugliano.info) (Bern, Friday March 8th, 2002 - 20:09)

HDR.ABF.ScopeOutputInterval  = fread(HDR.FILE.FID,1,'float'); % 174
HDR.ABF.EpisodeStartToStart  = fread(HDR.FILE.FID,1,'float'); % 178
HDR.ABF.RunStartToStart      = fread(HDR.FILE.FID,1,'float'); % 182
HDR.ABF.TrialStartToStart    = fread(HDR.FILE.FID,1,'float'); % 186
HDR.ABF.AverageCount         = fread(HDR.FILE.FID,1,'int');   % 190
HDR.ABF.ClockChange          = fread(HDR.FILE.FID,1,'int');   % 194
HDR.ABF.AutoTriggerStrategy  = fread(HDR.FILE.FID,1,'short'); % 198
%-----------------------------------------------------------------------------
% Display Parameters
HDR.ABF.DrawingStrategy      = fread(HDR.FILE.FID,1,'short'); % 200
HDR.ABF.TiledDisplay         = fread(HDR.FILE.FID,1,'short'); % 202
HDR.ABF.EraseStrategy        = fread(HDR.FILE.FID,1,'short'); % 204
HDR.ABF.DataDisplayMode      = fread(HDR.FILE.FID,1,'short'); % 206
HDR.ABF.DisplayAverageUpdate = fread(HDR.FILE.FID,1,'int');   % 208
HDR.ABF.ChannelStatsStrategy = fread(HDR.FILE.FID,1,'short'); % 212
HDR.ABF.CalculationPeriod    = fread(HDR.FILE.FID,1,'int');   % 214
HDR.ABF.SamplesPerTrace      = fread(HDR.FILE.FID,1,'int');   % 218
HDR.ABF.StartDisplayNum      = fread(HDR.FILE.FID,1,'int');   % 222
HDR.ABF.FinishDisplayNum     = fread(HDR.FILE.FID,1,'int');   % 226
HDR.ABF.MultiColor           = fread(HDR.FILE.FID,1,'short'); % 230
HDR.ABF.ShowPNRawData        = fread(HDR.FILE.FID,1,'short'); % 232
HDR.ABF.StatisticsPeriod     = fread(HDR.FILE.FID,1,'float'); % 234
HDR.ABF.StatisticsMeasurements=fread(HDR.FILE.FID,1,'int');   % 238
%-----------------------------------------------------------------------------
% Hardware Information
HDR.ABF.StatisticsSaveStrategy=fread(HDR.FILE.FID,1,'short'); % 242
HDR.ABF.ADCRange             = fread(HDR.FILE.FID,1,'float'); % 244
HDR.ABF.DACRange             = fread(HDR.FILE.FID,1,'float'); % 248
HDR.ABF.ADCResolution        = fread(HDR.FILE.FID,1,'int');   % 252
HDR.ABF.DACResolution        = fread(HDR.FILE.FID,1,'int');   % 256
%-----------------------------------------------------------------------------
% Environmental Information
HDR.ABF.ExperimentType       = fread(HDR.FILE.FID,1,'short'); % 260
HDR.ABF.x_AutosampleEnable   = fread(HDR.FILE.FID,1,'short'); % 262
HDR.ABF.x_AutosampleADCNum   = fread(HDR.FILE.FID,1,'short'); % 264
HDR.ABF.x_AutosampleInstrument=fread(HDR.FILE.FID,1,'short'); % 266
HDR.ABF.x_AutosampleAdditGain= fread(HDR.FILE.FID,1,'float'); % 268
HDR.ABF.x_AutosampleFilter   = fread(HDR.FILE.FID,1,'float'); % 272
HDR.ABF.x_AutosampleMembraneCapacitance=fread(HDR.FILE.FID,1,'float'); % 276
HDR.ABF.ManualInfoStrategy   = fread(HDR.FILE.FID,1,'short'); % 280
HDR.ABF.CellID1              = fread(HDR.FILE.FID,1,'float'); % 282
HDR.ABF.CellID2              = fread(HDR.FILE.FID,1,'float'); % 286
HDR.ABF.CellID3              = fread(HDR.FILE.FID,1,'float'); % 290
HDR.ABF.CreatorInfo          = fread(HDR.FILE.FID,16,'char'); % 16char % 294
HDR.ABF.x_FileComment        = fread(HDR.FILE.FID,56,'char'); % 56char % 310
HDR.ABF.Unused366            = fread(HDR.FILE.FID,12,'char'); % 12char % 366
%-----------------------------------------------------------------------------
% Multi-channel Information
HDR.ABF.ADCPtoLChannelMap    = fread(HDR.FILE.FID,16,'short');    % 378
HDR.ABF.ADCSamplingSeq       = fread(HDR.FILE.FID,16,'short');    % 410
HDR.ABF.ADCChannelName       = fread(HDR.FILE.FID,16*10,'char');  % 442
HDR.ABF.ADCUnits             = fread(HDR.FILE.FID,16*8,'char');   % 8char % 602
HDR.ABF.ADCProgrammableGain  = fread(HDR.FILE.FID,16,'float');    % 730
HDR.ABF.ADCDisplayAmplification=fread(HDR.FILE.FID,16,'float');   % 794
HDR.ABF.ADCDisplayOffset     = fread(HDR.FILE.FID,16,'float');    % 858
HDR.ABF.InstrumentScaleFactor= fread(HDR.FILE.FID,16,'float');    % 922
HDR.ABF.InstrumentOffset     = fread(HDR.FILE.FID,16,'float');    % 986
HDR.ABF.SignalGain           = fread(HDR.FILE.FID,16,'float');    % 1050
HDR.Off			     = fread(HDR.FILE.FID,16,'float');    % 1114
HDR.ABF.SignalLowpassFilter  = fread(HDR.FILE.FID,16,'float');    % 1178
HDR.ABF.SignalHighpassFilter = fread(HDR.FILE.FID,16,'float');    % 1242
HDR.ABF.DACChannelName       = fread(HDR.FILE.FID,4*10,'char');   % 1306
HDR.ABF.DACChannelUnits      = fread(HDR.FILE.FID,4*8,'char');    % 8char % 1346
HDR.ABF.DACScaleFactor       = fread(HDR.FILE.FID,4,'float');     % 1378
HDR.ABF.DACHoldingLevel      = fread(HDR.FILE.FID,4,'float');     % 1394
HDR.ABF.SignalType           = fread(HDR.FILE.FID,1,'short');     % 12char % 1410
HDR.ABF.Unused1412           = fread(HDR.FILE.FID,10,'char');     % 10char % 1412
%-----------------------------------------------------------------------------
% Synchronous Timer Outputs
HDR.ABF.OUTEnable            = fread(HDR.FILE.FID,1,'short');     % 1422
HDR.ABF.SampleNumberOUT1     = fread(HDR.FILE.FID,1,'short');     % 1424
HDR.ABF.SampleNumberOUT2     = fread(HDR.FILE.FID,1,'short');     % 1426
HDR.ABF.FirstEpisodeOUT      = fread(HDR.FILE.FID,1,'short');     % 1428
HDR.ABF.LastEpisodeOUT       = fread(HDR.FILE.FID,1,'short');     % 1430
HDR.ABF.PulseSamplesOUT1     = fread(HDR.FILE.FID,1,'short');     % 1432
HDR.ABF.PulseSamplesOUT2     = fread(HDR.FILE.FID,1,'short');     % 1434
%-----------------------------------------------------------------------------
% Epoch Waveform and Pulses
HDR.ABF.DigitalEnable        = fread(HDR.FILE.FID,1,'short');     % 1436
HDR.ABF.x_WaveformSource     = fread(HDR.FILE.FID,1,'short');     % 1438
HDR.ABF.ActiveDACChannel     = fread(HDR.FILE.FID,1,'short');     % 1440
HDR.ABF.x_InterEpisodeLevel  = fread(HDR.FILE.FID,1,'short');     % 1442
HDR.ABF.x_EpochType          = fread(HDR.FILE.FID,10,'short');    % 1444
HDR.ABF.x_EpochInitLevel     = fread(HDR.FILE.FID,10,'float');    % 1464
HDR.ABF.x_EpochLevelInc      = fread(HDR.FILE.FID,10,'float');    % 1504
HDR.ABF.x_EpochInitDuration  = fread(HDR.FILE.FID,10,'short');    % 1544
HDR.ABF.x_EpochDurationInc   = fread(HDR.FILE.FID,10,'short');    % 1564
HDR.ABF.DigitalHolding       = fread(HDR.FILE.FID,1,'short');     % 1584
HDR.ABF.DigitalInterEpisode  = fread(HDR.FILE.FID,1,'short');     % 1586
HDR.ABF.DigitalValue         = fread(HDR.FILE.FID,10,'short');    % 1588
HDR.ABF.Unavailable1608      = fread(HDR.FILE.FID,4,'char');      % 1608
HDR.ABF.Unused1612           = fread(HDR.FILE.FID,8,'char');      % 8char % 1612
%-----------------------------------------------------------------------------
% DAC Output File
HDR.ABF.x_DACFileScale       = fread(HDR.FILE.FID,1,'float');     % 1620
HDR.ABF.x_DACFileOffset      = fread(HDR.FILE.FID,1,'float');     % 1624
HDR.ABF.Unused1628           = fread(HDR.FILE.FID,2,'char');      % 2char % 1628
HDR.ABF.x_DACFileEpisodeNum  = fread(HDR.FILE.FID,1,'short');     % 1630
HDR.ABF.x_DACFileADCNum      = fread(HDR.FILE.FID,1,'short');     % 1632
HDR.ABF.x_DACFileName        = fread(HDR.FILE.FID,12,'char');     % 12char % 1634
HDR.ABF.DACFilePath=fread(HDR.FILE.FID,60,'char');                % 60char % 1646
HDR.ABF.Unused1706=fread(HDR.FILE.FID,12,'char');                 % 12char % 1706
%-----------------------------------------------------------------------------
% Conditioning Pulse Train
HDR.ABF.x_ConditEnable       = fread(HDR.FILE.FID,1,'short');     % 1718
HDR.ABF.x_ConditChannel      = fread(HDR.FILE.FID,1,'short');     % 1720
HDR.ABF.x_ConditNumPulses    = fread(HDR.FILE.FID,1,'int');       % 1722
HDR.ABF.x_BaselineDuration   = fread(HDR.FILE.FID,1,'float');     % 1726
HDR.ABF.x_BaselineLevel      = fread(HDR.FILE.FID,1,'float');     % 1730
HDR.ABF.x_StepDuration       = fread(HDR.FILE.FID,1,'float');     % 1734
HDR.ABF.x_StepLevel          = fread(HDR.FILE.FID,1,'float');     % 1738
HDR.ABF.x_PostTrainPeriod    = fread(HDR.FILE.FID,1,'float');     % 1742
HDR.ABF.x_PostTrainLevel     = fread(HDR.FILE.FID,1,'float');     % 1746
HDR.ABF.Unused1750           = fread(HDR.FILE.FID,12,'char');     % 12char % 1750
%-----------------------------------------------------------------------------
% Variable Parameter User List
HDR.ABF.x_ParamToVary        = fread(HDR.FILE.FID,1,'short');     % 1762
HDR.ABF.x_ParamValueList     = fread(HDR.FILE.FID,80,'char');     % 80char % 1764
%-----------------------------------------------------------------------------
% Statistics Measurement
HDR.ABF.AutopeakEnable       = fread(HDR.FILE.FID,1,'short'); % 1844
HDR.ABF.AutopeakPolarity     = fread(HDR.FILE.FID,1,'short'); % 1846
HDR.ABF.AutopeakADCNum       = fread(HDR.FILE.FID,1,'short'); % 1848
HDR.ABF.AutopeakSearchMode   = fread(HDR.FILE.FID,1,'short'); % 1850
HDR.ABF.AutopeakStart        = fread(HDR.FILE.FID,1,'int');   % 1852
HDR.ABF.AutopeakEnd          = fread(HDR.FILE.FID,1,'int');   % 1856
HDR.ABF.AutopeakSmoothing    = fread(HDR.FILE.FID,1,'short'); % 1860
HDR.ABF.AutopeakBaseline     = fread(HDR.FILE.FID,1,'short'); % 1862
HDR.ABF.AutopeakAverage      = fread(HDR.FILE.FID,1,'short'); % 1864
HDR.ABF.Unavailable1866      = fread(HDR.FILE.FID,2,'char');  % 1866
HDR.ABF.AutopeakBaselineStart= fread(HDR.FILE.FID,1,'int');   % 1868
HDR.ABF.AutopeakBaselineEnd  = fread(HDR.FILE.FID,1,'int');   % 1872
HDR.ABF.AutopeakMeasurements = fread(HDR.FILE.FID,1,'int');   % 1876
%-----------------------------------------------------------------------------
% Channel Arithmetic
HDR.ABF.ArithmeticEnable     = fread(HDR.FILE.FID,1,'short'); % 1880
HDR.ABF.ArithmeticUpperLimit = fread(HDR.FILE.FID,1,'float'); % 1882
HDR.ABF.ArithmeticLowerLimit = fread(HDR.FILE.FID,1,'float'); % 1886
HDR.ABF.ArithmeticADCNumA    = fread(HDR.FILE.FID,1,'short'); % 1890
HDR.ABF.ArithmeticADCNumB    = fread(HDR.FILE.FID,1,'short'); % 1892
HDR.ABF.ArithmeticK1         = fread(HDR.FILE.FID,1,'float'); % 1894
HDR.ABF.ArithmeticK2         = fread(HDR.FILE.FID,1,'float'); % 1898
HDR.ABF.ArithmeticK3         = fread(HDR.FILE.FID,1,'float'); % 1902
HDR.ABF.ArithmeticK4         = fread(HDR.FILE.FID,1,'float'); % 1906
HDR.ABF.ArithmeticOperator   = fread(HDR.FILE.FID,2,'char');  % 2char % 1910
HDR.ABF.ArithmeticUnits      = fread(HDR.FILE.FID,8,'char');  % 8char % 1912
HDR.ABF.ArithmeticK5         = fread(HDR.FILE.FID,1,'float'); % 1920
HDR.ABF.ArithmeticK6         = fread(HDR.FILE.FID,1,'float'); % 1924
HDR.ABF.ArithmeticExpression = fread(HDR.FILE.FID,1,'short'); % 1928
HDR.ABF.Unused1930           = fread(HDR.FILE.FID,2,'char');  % 2char % 1930
%-----------------------------------------------------------------------------
% On-line Subtraction
HDR.ABF.x_PNEnable           = fread(HDR.FILE.FID,1,'short'); % 1932
HDR.ABF.PNPosition           = fread(HDR.FILE.FID,1,'short'); % 1934
HDR.ABF.x_PNPolarity         = fread(HDR.FILE.FID,1,'short'); % 1936
HDR.ABF.PNNumPulses          = fread(HDR.FILE.FID,1,'short'); % 1938
HDR.ABF.x_PNADCNum           = fread(HDR.FILE.FID,1,'short'); % 1940
HDR.ABF.x_PNHoldingLevel     = fread(HDR.FILE.FID,1,'float'); % 1942
HDR.ABF.PNSettlingTime       = fread(HDR.FILE.FID,1,'float'); % 1946
HDR.ABF.PNInterpulse         = fread(HDR.FILE.FID,1,'float'); % 1950
HDR.ABF.Unused1954           = fread(HDR.FILE.FID,12,'char'); % 12char % 1954
%-----------------------------------------------------------------------------
% Unused Space at End of Header Block
HDR.ABF.x_ListEnable         = fread(HDR.FILE.FID,1,'short'); % 1966
HDR.ABF.BellEnable           = fread(HDR.FILE.FID,2,'short'); % 1968
HDR.ABF.BellLocation         = fread(HDR.FILE.FID,2,'short'); % 1972
HDR.ABF.BellRepetitions      = fread(HDR.FILE.FID,2,'short'); % 1976
HDR.ABF.LevelHysteresis      = fread(HDR.FILE.FID,1,'int');   % 1980
HDR.ABF.TimeHysteresis       = fread(HDR.FILE.FID,1,'int');   % 1982
HDR.ABF.AllowExternalTags    = fread(HDR.FILE.FID,1,'short'); % 1986
HDR.ABF.LowpassFilterType    = fread(HDR.FILE.FID,16,'char'); % 1988
HDR.ABF.HighpassFilterType   = fread(HDR.FILE.FID,16,'char');% 2004
HDR.ABF.AverageAlgorithm     = fread(HDR.FILE.FID,1,'short'); % 2020
HDR.ABF.AverageWeighting     = fread(HDR.FILE.FID,1,'float'); % 2022
HDR.ABF.UndoPromptStrategy   = fread(HDR.FILE.FID,1,'short'); % 2026
HDR.ABF.TrialTriggerSource   = fread(HDR.FILE.FID,1,'short'); % 2028
HDR.ABF.StatisticsDisplayStrategy= fread(HDR.FILE.FID,1,'short'); % 2030
HDR.ABF.Unused2032           = fread(HDR.FILE.FID,16,'char'); % 2032

%-----------------------------------------------------------------------------
% File Structure 2
HDR.ABF.DACFilePtr           = fread(HDR.FILE.FID,2,'int'); % 2048
HDR.ABF.DACFileNumEpisodes   = fread(HDR.FILE.FID,2,'int'); % 2056
HDR.ABF.Unused2              = fread(HDR.FILE.FID,10,'char');%2064
%-----------------------------------------------------------------------------
% Multi-channel Information 2
HDR.ABF.DACCalibrationFactor = fread(HDR.FILE.FID,4,'float'); % 2074
HDR.ABF.DACCalibrationOffset = fread(HDR.FILE.FID,4,'float'); % 2090
HDR.ABF.Unused7              = fread(HDR.FILE.FID,190,'char');% 2106
%-----------------------------------------------------------------------------
% Epoch Waveform and Pulses 2
HDR.ABF.WaveformEnable       = fread(HDR.FILE.FID,2,'short'); % 2296
HDR.ABF.WaveformSource       = fread(HDR.FILE.FID,2,'short'); % 2300
HDR.ABF.InterEpisodeLevel    = fread(HDR.FILE.FID,2,'short'); % 2304
HDR.ABF.EpochType            = fread(HDR.FILE.FID,10*2,'short');% 2308
HDR.ABF.EpochInitLevel       = fread(HDR.FILE.FID,10*2,'float');% 2348
HDR.ABF.EpochLevelInc        = fread(HDR.FILE.FID,10*2,'float');% 2428
HDR.ABF.EpochInitDuration    = fread(HDR.FILE.FID,10*2,'int');  % 2508
HDR.ABF.EpochDurationInc     = fread(HDR.FILE.FID,10*2,'int');  % 2588
HDR.ABF.Unused9              = fread(HDR.FILE.FID,40,'char');   % 2668
%-----------------------------------------------------------------------------
% DAC Output File 2
HDR.ABF.DACFileScale         = fread(HDR.FILE.FID,2,'float');     % 2708
HDR.ABF.DACFileOffset        = fread(HDR.FILE.FID,2,'float');     % 2716
HDR.ABF.DACFileEpisodeNum    = fread(HDR.FILE.FID,2,'int');       % 2724
HDR.ABF.DACFileADCNum        = fread(HDR.FILE.FID,2,'short');     % 2732
HDR.ABF.DACFilePath          = fread(HDR.FILE.FID,2*256,'char');  % 2736
HDR.ABF.Unused10             = fread(HDR.FILE.FID,12,'char');     % 3248
%-----------------------------------------------------------------------------
% Conditioning Pulse Train 2
HDR.ABF.ConditEnable         = fread(HDR.FILE.FID,2,'short');     % 3260
HDR.ABF.ConditNumPulses      = fread(HDR.FILE.FID,2,'int');       % 3264
HDR.ABF.BaselineDuration     = fread(HDR.FILE.FID,2,'float');     % 3272
HDR.ABF.BaselineLevel        = fread(HDR.FILE.FID,2,'float');     % 3280
HDR.ABF.StepDuration         = fread(HDR.FILE.FID,2,'float');     % 3288
HDR.ABF.StepLevel            = fread(HDR.FILE.FID,2,'float');     % 3296
HDR.ABF.PostTrainPeriod      = fread(HDR.FILE.FID,2,'float');     % 3304
HDR.ABF.PostTrainLevel       = fread(HDR.FILE.FID,2,'float');     % 3312
HDR.ABF.Unused11             = fread(HDR.FILE.FID,2,'short');     % 3320
HDR.ABF.Unused11             = fread(HDR.FILE.FID,36,'char');     % 3324
%-----------------------------------------------------------------------------
% Variable Parameter User List 2
HDR.ABF.ULEnable             = fread(HDR.FILE.FID,4,'short');     % 3360
HDR.ABF.ULParamToVary        = fread(HDR.FILE.FID,4,'short');     % 3368
HDR.ABF.ULParamValueList     = fread(HDR.FILE.FID,4*256,'char');  % 3376
HDR.ABF.Unused11             = fread(HDR.FILE.FID,56,'char');     % 4400
%-----------------------------------------------------------------------------
% On-line Subtraction 2
HDR.ABF.PNEnable             = fread(HDR.FILE.FID,2,'short');     % 4456
HDR.ABF.PNPolarity           = fread(HDR.FILE.FID,2,'short');     % 4460
HDR.ABF.PNADCNum             = fread(HDR.FILE.FID,2,'short');     % 4464
HDR.ABF.PNHoldingLevel       = fread(HDR.FILE.FID,2,'float');     % 4468
HDR.ABF.Unused15             = fread(HDR.FILE.FID,36,'char');     % 4476
%-----------------------------------------------------------------------------
% Environmental Information 2
HDR.ABF.TelegraphEnable      = fread(HDR.FILE.FID,16,'short');     % 4512
HDR.ABF.TelegraphInstrument  = fread(HDR.FILE.FID,16,'short');     % 4544
HDR.ABF.TelegraphAdditGain   = fread(HDR.FILE.FID,16,'float');     % 4576
HDR.ABF.TelegraphFilter      = fread(HDR.FILE.FID,16,'float');     % 4640
HDR.ABF.TelegraphMembraneCap = fread(HDR.FILE.FID,16,'float');     % 4704
HDR.ABF.TelegraphMode        = fread(HDR.FILE.FID,16,'short');     % 4768
HDR.ABF.ManualTelegraphStrategy= fread(HDR.FILE.FID,16,'short');   % 4800
HDR.ABF.AutoAnalyseEnable    = fread(HDR.FILE.FID,1,'short');      % 4832
HDR.ABF.AutoAnalysisMacroName= fread(HDR.FILE.FID,64,'char');      % 4834
HDR.ABF.ProtocolPath         = fread(HDR.FILE.FID,256,'char');     % 4898
HDR.ABF.FileComment          = fread(HDR.FILE.FID,128,'char');     % 5154
HDR.ABF.Unused6              = fread(HDR.FILE.FID,128,'char');     % 5282
HDR.ABF.Unused2048           = fread(HDR.FILE.FID,734,'char');     % 5410
%
%-----------------------------------------------------------------------------
%

		HDR.Cal = (HDR.ABF.ADCRange / (HDR.ABF.ADCResolution * HDR.ABF.x_AutosampleAdditGain))./ (HDR.ABF.InstrumentScaleFactor .* HDR.ABF.ADCProgrammableGain .* HDR.ABF.SignalGain);
				
		HDR.Calib = sparse([HDR.Off(1:HDR.NS)'; diag(HDR.Cal(1:HDR.NS))]);

		status = fseek(HDR.FILE.FID,HDR.HeadLen,'bof');
		HDR.FILE.POS = 0; 
		%HDR.FILE.OPEN = 1; 

		HDR.data = fread(HDR.FILE.FID,[HDR.NS,HDR.NRec*HDR.SPR],gdfdatatype(HDR.GDFTYP))';
		HDR.TYPE = 'native';
		fclose(HDR.FILE.FID);
        end


elseif strcmp(HDR.TYPE,'ATF'),  % axon text file 
        if any(PERMISSION=='r'),
                HDR.FILE.FID = fopen(HDR.FileName,'rt','ieee-le');
                t = fgetl(HDR.FILE.FID);
                t = str2double(fgetl(HDR.FILE.FID));
                HDR.ATF.NoptHdr = t(1);
                HDR.ATF.NS = t(2);
                HDR.ATF.NormalizationFactor = [];
                t = fgetl(HDR.FILE.FID);
                while any(t=='=')
                        [f,t]=strtok(t,[34,61]);        %  "= 
                        [c,t]=strtok(t,[34,61]);        %  "= 
                        if strfind(f,'NormalizationFactor:')
                                [t1, t2] = strtok(f,':');
                                [f] = strtok(t2,':');
                                HDR.ATF.NormalizationFactor = setfield(HDR.ATF.NormalizationFactor,f,str2double(c));        
                        else
                                HDR.ATF = setfield(HDR.ATF,f,c);        
                        end
                        t = fgetl(HDR.FILE.FID);
                end;
                k = 0;
                HDR.Label = {};
                while ~isempty(t),
                        k = k + 1;
                        [HDR.Label{k,1},t] = strtok(t,[9,34]);    % ", TAB
                end
                HDR.HeadLen = ftell(HDR.FILE.FID);
                if isfield(HDR.ATF,'DateTime');
                        tmp = HDR.ATF.DateTime;
                        tmp(tmp=='/' | tmp==':')=' ';
                        HDR.T0 = str2double(tmp);
                end;
                HDR.FILE.OPEN = 1; 
        end

        
elseif strcmp(HDR.TYPE,'BIFF'),
	try, 
                [HDR.TFM.S,HDR.TFM.E] = xlsread(HDR.FileName,'Beat-To-Beat');
                %if size(HDR.TFM.S,1)+1==size(HDR.TFM.E,1)
                if ~isnan(HDR.TFM.S(1,1)) & ~isempty(HDR.TFM.E{1,1})
		        fprintf('Warning: XLSREAD-BUG has occured in file %s.\n',HDR.FileName);
                        HDR.TFM.S = [repmat(NaN,1,size(HDR.TFM.S,2));HDR.TFM.S];
                end;
                
                HDR.TYPE = 'TFM_EXCEL_Beat_to_Beat'; 
                %HDR.Patient.Name = [HDR.TFM.E{2,3},' ', HDR.TFM.E{2,4}];
                HDR.Patient.Birthday = datevec(HDR.TFM.S(2,5)-1);
                HDR.Patient.Age = datevec(HDR.TFM.S(2,1)-HDR.TFM.S(2,5));
                HDR.Patient.Sex = HDR.TFM.E{2,6};
                HDR.Patient.Height = HDR.TFM.S(2,7);
                HDR.Patient.Weight = HDR.TFM.S(2,8);
                HDR.Patient.Surface = HDR.TFM.S(2,9);
                HDR.Patient.BMI = HDR.TFM.S(2,8)*HDR.TFM.S(2,7)^-2*1e4;
                HDR.TFM.VERSION = HDR.TFM.E{2,11};
                HDR.T0 = datevec(datenum('30-Dec-1899')+HDR.TFM.S(2,1)+HDR.TFM.S(2,2));
        catch
	end; 	

        if strcmp(HDR.TYPE, 'TFM_EXCEL_Beat_to_Beat');
                if ~isempty(strfind(HDR.TFM.E{3,1},'---'))
                        HDR.TFM.S(3,:) = [];    
                        HDR.TFM.E(3,:) = [];    
                end;
                
                HDR.Label   = strvcat(HDR.TFM.E{4,:});
                HDR.PhysDim = strvcat(HDR.TFM.E{5,:});
           
                HDR.TFM.S = HDR.TFM.S(6:end,:);
                HDR.TFM.E = HDR.TFM.E(6:end,:);
		
                ix = find(isnan(HDR.TFM.S(:,2)) & ~isnan(HDR.TFM.S(:,1)));
                HDR.EVENT.Desc = HDR.TFM.E(ix,2);
                HDR.EVENT.POS  = ix(:);
                HDR.EVENT.TYP  = zeros(size(HDR.EVENT.POS));
                
		[HDR.SPR,HDR.NS] = size(HDR.TFM.S);
                if any(CHAN),
			HDR.TFM.S = HDR.TFM.S(:,CHAN);
			HDR.TFM.E = HDR.TFM.E(:,CHAN);
		end;
		HDR.NRec = 1;
		HDR.THRESHOLD  = repmat([0,NaN],HDR.NS,1); 	% Underflow Detection 
		HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,1);
        end;


elseif strncmp(HDR.TYPE,'XML',3),
        if any(PERMISSION=='r'),
                fid = fopen(HDR.FileName,'rb','ieee-le');
                if strcmp(HDR.TYPE,'XML-UTF16'),
                        magic = char(fread(fid,1,'uint16'));
                        HDR.XML = char(fread(fid,[1,inf],'uint16'));
                elseif strcmp(HDR.TYPE,'XML-UTF8'),
                        HDR.XML = char(fread(fid,[1,inf],'char'));
                end;
                fclose(fid);
                HDR.FILE.FID = fid;
                if 1, try,
                        XML = xmltree(HDR.XML);
                        XML = convert(XML);
                        HDR.XML  =  XML; 
			HDR.TYPE = 'XML';
                catch
                        fprintf(HDR.FILE.stderr,'ERROR SOPEN (XML): XML-toolbox missing or invalid XML file.\n');
                        return;
                end;
                end;
		
		
                try,    % SierraECG  1.03  *.open.xml from PHILIPS
                        HDR.SampleRate = str2double(HDR.XML.dataacquisition.signalcharacteristics.samplingrate);
                        HDR.NS  = str2double(HDR.XML.dataacquisition.signalcharacteristics.numberchannelsvalid);
                        HDR.Cal = str2double(HDR.XML.reportinfo.reportgain.amplitudegain.overallgain);
                        HDR.PhysDim = 'uV';
                        HDR.Filter.HighPass = str2double(HDR.XML.reportinfo.reportbandwidth.highpassfiltersetting);
                        HDR.Filter.LowPass  = str2double(HDR.XML.reportinfo.reportbandwidth.lowpassfiltersetting);
                        HDR.Filter.Notch    = str2double(HDR.XML.reportinfo.reportbandwidth.notchfiltersetting);
                        
                        t = HDR.XML.reportinfo.reportformat.waveformformat.mainwaveformformat;
                        k = 0; 
                        HDR.Label=[];
                        while ~isempty(t),
                                [s,t] = strtok(t,' ');
                                k = k+1; 
                                HDR.Label{k, 1} = [s,' '];
                        end;
                        
                        HDR.VERSION = HDR.XML.documentinfo.documentversion;
                        HDR.TYPE = HDR.XML.documentinfo.documenttype;
                catch
                        
                try,    % FDA-XML Format
                        tmp   = HDR.XML.component.series.derivation;
                        if isfield(tmp,'Series');
                                tmp = tmp.Series.component.sequenceSet.component;
                        else    % Dovermed.CO.IL version of format
                                tmp = tmp.derivedSeries.component.sequenceSet.component;
                        end;
                        HDR.NS = length(tmp)-1;
                        HDR.NRec = 1; 
                        HDR.Cal = 1;
                        HDR.PhysDim = ' ';
                        HDR.SampleRate = 1;
                        HDR.TYPE = 'XML-FDA';     % that's an FDA XML file 
                catch
                        fprintf(HDR.FILE.stderr,'Warning SOPEN (XML): File %s is not supported.\n',HDR.FileName);
                        return;
                end;
                end
                
                HDR.Calib = sparse(2:HDR.NS+1,1:HDR.NS,HDR.Cal);
                HDR.FILE.OPEN = 1;
                HDR.FILE.POS  = 0;
        end;
        
        
elseif strncmp(HDR.TYPE,'IMAGE:',6),
	% forward call to IOPEN
        HDR = iopen(HDR);
	return;


elseif strcmp(HDR.TYPE,'unknown'),
        if HDR.FLAG.ASCII, 
        	s = HDR.s; 
                if strcmpi(HDR.FILE.Ext,'DAT') 
                	[NUM, STATUS,STRARRAY] = str2double(char(s));
                        if (size(NUM,2)<4) & ~any(any(STATUS))
                                HDR.Label = strvcat(STRARRAY(:,1));
                                r2 = sum(NUM(:,2:3).^2,2);
                                HDR.ELEC.XYZ = [NUM(:,2:3),sqrt(max(r2)-r2)]; 
                                HDR.CHAN  = NUM(:,1); 
                                HDR.TYPE  = 'ELPOS'; 
                        elseif (size(NUM,2)==4) & ~any(any(STATUS))
                                HDR.Label = strvcat(STRARRAY(:,1));
                                HDR.ELEC.XYZ  = NUM(:,2:4); 
                                HDR.ELEC.CHAN = NUM(:,1); 
                                HDR.TYPE  = 'ELPOS'; 
                        elseif (size(NUM,2)==4) & ~any(any(STATUS(:,[1,3:4])))
                                HDR.Label = strvcat(STRARRAY(:,2));
                                r2 = sum(NUM(:,3:4).^2,2);
                                HDR.ELEC.XYZ = [NUM(:,3:4),sqrt(max(r2)-r2)]; 
                                HDR.CHAN  = NUM(:,1); 
                                HDR.TYPE  = 'ELPOS'; 
                        elseif (size(NUM,2)==5) & ~any(any(STATUS(:,3:5)))
                                HDR.Label = strvcat(STRARRAY(:,1));
                                HDR.ELEC.XYZ  = NUM(:,3:5); 
                                HDR.TYPE  = 'ELPOS'; 
                        end;
                        
                elseif strncmp(s,'NumberPositions',15) & strcmpi(HDR.FILE.Ext,'elc');  % Polhemus 
                        K = 0; 
                        [tline, s] = strtok(s, [10,13]);
                        while ~isempty(s),
                                [num, stat, strarray] = str2double(tline); 
                                if strcmp(strarray{1},'NumberPositions')
                                        NK = num(2); 
                                elseif strcmp(strarray{1},'UnitPosition')
                                        HDR.ELEC.PositionUnit = strarray{2};
                                elseif strcmp(strarray{1},'Positions')
                                        ix = strfind(s,'Labels');
                                        ix = min([ix-1,length(s)]);
                                        [num, stat, strarray] = str2double(s(1:ix));
                                        s(1:ix) = [];
                                        if ~any(any(stat))
                                                HDR.ELEC.XYZ = num*[0,-1,0;1,0,0;0,0,1]; 
                                                HDR.TYPE = 'ELPOS'; 
                                        end;
                                elseif strcmp(strarray{1},'Labels')
                                        [tline, s] = strtok(s, [10,13]); 
                                        [num, stat, strarray] = str2double(tline);
                                        HDR.Label = strarray';
                                end
                                [tline, s] = strtok(s, [10,13]);
                        end;
                        
                elseif strncmp(s,'Site',4) & strcmpi(HDR.FILE.Ext,'txt'); 
                        [line1, s] = strtok(s, [10,13]); 
                        s(s==',') = '.';
                        [NUM, STATUS, STRARRAY] = str2double(s,[9,32]);
                        if (size(NUM,2)==3) & ~any(any(STATUS(:,2:3)))
                                HDR.Label = strvcat(STRARRAY(:,1));
                                Theta     = abs(NUM(:,2))*pi/180; 
                                Phi       = NUM(:,3)*pi/180 + pi*(NUM(:,2)<0); 
                                HDR.ELEC.XYZ = [sin(Theta).*cos(Phi),sin(Theta).*sin(Phi),cos(Theta)]; 
                                HDR.ELEC.R   = 1; 
                                HDR.TYPE     = 'ELPOS'; 
                        elseif (size(NUM,2)==4) & ~any(any(STATUS(:,2:4)))
                                HDR.Label = strvcat(STRARRAY(:,1));
                                HDR.ELEC.XYZ = NUM(:,2:4); 
                                HDR.TYPE  = 'ELPOS'; 
                        end;
                        
                elseif strcmpi(HDR.FILE.Ext,'elp')
                        [line1,s]=strtok(s,[10,13]);
                        [NUM, STATUS,STRARRAY] = str2double(char(s));
                        if size(NUM,2)==3,
                                if ~any(any(STATUS(:,2:3)))
                                        HDR.Label = strvcat(STRARRAY(:,1));
	                                Theta = NUM(:,2)*pi/180; 
	                                Phi   = NUM(:,3)*pi/180; 
	                                HDR.ELEC.XYZ = [sin(Theta).*cos(Phi),sin(Theta).*sin(Phi),cos(Theta)]; 
	                                HDR.ELEC.R   = 1; 
                                        HDR.TYPE = 'ELPOS'; 
                                end;
                        elseif size(NUM,2)==4,
                                if ~any(any(STATUS(:,3:4)))
                                        HDR.Label = strvcat(STRARRAY(:,2));
	                                Theta = NUM(:,2)*pi/180; 
	                                Phi   = NUM(:,3)*pi/180; 
	                                HDR.ELEC.XYZ = [sin(Theta).*cos(Phi),sin(Theta).*sin(Phi),cos(Theta)]; 
	                                HDR.ELEC.R   = 1; 
                                        HDR.ELEC.CHAN = NUM(:,1); 
                                        HDR.TYPE = 'ELPOS'; 
                                end;
                        end;
                        
                elseif strcmpi(HDR.FILE.Ext,'ced')
                        [line1,s]=strtok(char(s),[10,13]);
                        [NUM, STATUS,STRARRAY] = str2double(char(s));
                        if ~any(any(STATUS(:,[1,5:7])))
                                HDR.Label = strvcat(STRARRAY(:,2));
                                HDR.ELEC.XYZ  = NUM(:,5:7)*[0,1,0;-1,0,0;0,0,1]; 
                                HDR.ELEC.CHAN = NUM(:,1); 
                                HDR.TYPE  = 'ELPOS'; 
                        end;
                        
                elseif (strcmpi(HDR.FILE.Ext,'loc') | strcmpi(HDR.FILE.Ext,'locs'))
%                        [line1,s]=strtok(char(s),[10,13]);
                        [NUM, STATUS,STRARRAY] = str2double(char(s));
                        if ~any(any(STATUS(:,1:3)))
                                HDR.Label = strvcat(STRARRAY(:,4));
                                HDR.CHAN  = NUM(:,1); 
                                Phi       = NUM(:,2)/180*pi; 
                                Theta     = NUM(:,3)*pi;
                                HDR.ELEC.XYZ = [sin(Theta).*sin(Phi),sin(Theta).*cos(Phi),cos(Theta)]; 
	                        HDR.TYPE  = 'ELPOS'; 
                        end;
                        
                elseif strcmpi(HDR.FILE.Ext,'sfp')
                        [NUM, STATUS,STRARRAY] = str2double(char(s));
                        if ~any(any(STATUS(:,2:4)))
                                HDR.Label    = strvcat(STRARRAY(:,1));
                                HDR.ELEC.XYZ = NUM(:,2:4); 
                                HDR.TYPE     = 'ELPOS'; 
                        end;

                elseif strcmpi(HDR.FILE.Ext,'xyz')
                        [NUM, STATUS,STRARRAY] = str2double(char(s));
                        if ~any(any(STATUS(:,2:4)))
                                HDR.Label    = strvcat(STRARRAY(:,5));
                                HDR.ELEC.CHAN= NUM(:,1); 
                                HDR.ELEC.XYZ = NUM(:,2:4); 
                                HDR.TYPE     = 'ELPOS'; 
                        end;
        	end;
	else
		%HDR.ERROR.message = sprintf('ERROR SOPEN: File %s could not be opened - unknown type.\n',HDR.FileName);
		%fprintf(HDR.FILE.stderr,'ERROR SOPEN: File %s could not be opened - unknown type.\n',HDR.FileName);
        end; 
        return;
        
else
        %fprintf(HDR.FILE.stderr,'SOPEN does not support your data format yet. Contact <a.schloegl@ieee.org> if you are interested in this feature.\n');
        HDR.FILE.FID = -1;	% this indicates that file could not be opened. 
        return;
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	General Postprecessing for all formats of Header information 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set FLAGS 
HDR.FLAG.UCAL = ~isempty(strfind(MODE,'UCAL'));   % FLAG for UN-CALIBRATING
HDR.FLAG.OVERFLOWDETECTION = isempty(strfind(upper(MODE),'OVERFLOWDETECTION:OFF'));
%if ~isempty(strfind(upper(MODE),'OVERFLOWDETECTION:ON')) & ~isfield(HDR,'THRESHOLD'),
if HDR.FLAG.OVERFLOWDETECTION & ~isfield(HDR,'THRESHOLD'),
        fprintf(HDR.FILE.stderr,'Warning SOPEN: OVERFLOWDETECTION not supported because of missing THRESHOLD.\n');
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
        if isfield(HDR, 'Calib'),
          HDR.Calib = HDR.Calib*sparse([ReRefMx; zeros(HDR.NS-sz(1),sz(2))]);
        
          HDR.InChanSelect = find(any(HDR.Calib(2:HDR.NS+1,:),2));
          HDR.Calib = sparse(HDR.Calib([1;1+HDR.InChanSelect(:)],:));
          if strcmp(HDR.TYPE,'native')
            HDR.data = HDR.data(:,HDR.InChanSelect);
          end
        else
          HDR.Calib= sparse([ones(1,HDR.NS); eye(HDR.NS)]);
          HDR.InChanSelect = 1:HDR.NS;
        end;
end;
