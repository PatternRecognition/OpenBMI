function [clab, scale, fs, endian, len, PRS,ERS,out_ELS]= ...
    readKEHHeader(hdrName, float_format)
%[clab, scale, fs, endian, len, PRS, ERS, ELS]= readKEHHeader(hdrName, float_format)
%
% IN   hrdName - name of header file (no extension),
%                relative to EEG_RAW_DIR unless beginning with '/'
%
% OUT  clab    - channel labels (cell array)
%      scale   - scaling factors for each channel
%      fs      - sampling interval of raw data
%      endian  - byte ordering: 'l' little or 'b' big
%      len     - length of the data set in seconds
%      PRS     - Project Record Structure
%      ERS     - EEG Record Structure
%
% Attention! This appears not to work on some platforms, since data types like 
% uint16 etc. are represented differently.
%
% GLOBZ  EEG_RAW_DIR

if ~exist('float_format', 'var'), float_format=[]; end

if hdrName(1)==filesep,
  fullName= hdrName;
else
  global EEG_RAW_DIR
  fullName= [EEG_RAW_DIR hdrName];
end

fid= fopen([fullName '.eeg'], 'r'); 
if fid==-1, error(sprintf('%s.eeg not found', fullName)); end

% File-Typ-Kennung:
dum = fread(fid,4);


% Project_Record_Struktur
PRS = struct('Name','');
PRS.Name = stringread(fid,80);
PRS.Vorname = stringread(fid,64);
PRS.EEG_Nummer = stringread(fid,6);
PRS.Messplatz_Kennung = stringread(fid,10);
PRS.Geburtsdatum = datestr(datenum(1899,12,30)+fread(fid,1,'double'));
PRS.ID =  stringread(fid,16);
PRS.Interiktal_Lang = fread(fid,1,'uint16');
PRS.Interiktal_Intervall = fread(fid,1,'uint16');
PRS.Iktal_nach_Marker = fread(fid,1,'uint16');
PRS.Iktal_vor_Marker = fread(fid,1,'uint16');
PRS.Routine_Lang = fread(fid,1,'uint16');
PRS.AFile_Nummer = fread(fid,1,'uint16');
PRS.File_Startzeit_mit_Datum = datestr(datenum(1899,12,30)+fread(fid,1,'double'));
PRS.Kassetten_Nr = fread(fid,1,'uint32');
PRS.Kanalzahl_gespeichert = fread(fid,1,'uint16');
PRS.Video_Modus = fread(fid,1,'uint16');
PRS.Video_Start_Sample = fread(fid,1,'uint32');
PRS.Video_Start_Frame = fread(fid,1,'uint32');
PRS.Gesamtanzahl_HEB = fread(fid,1,'uint16');
PRS.Gesamtanzahl_UEB = fread(fid,1,'uint16');
PRS.Seekposition_Datenbeginn = fread(fid,1,'uint32');
% up to here: 232 bytes.

% EEG_Record_Struktur
ERS = struct('fadc',[]);
ERS.fadc = fread(fid,1,'double');
ERS.fdig = fread(fid,1,'double');
ERS.DigitSpannung_in_Mikrovolt = fread(fid,1,'double');
ERS.Max_Kanalzahl = fread(fid,1,'uint16');
ERS.Datenpfad = stringread(fid,80);
ERS.Kassetten_Lang = fread(fid,1,'uint16');
ERS.Anzahl_AFile = fread(fid,1,'uint16');
fs = ERS.fdig;
% up to here: 342 bytes.
%numChans = 128;
numChans = PRS.Kanalzahl_gespeichert;
ELS = struct('Bezeichnung','');
clab = {};
tec_chan_ind=0;
chan_ind = 0;
for ii = 1:PRS.Gesamtanzahl_UEB
  % for the moment:skip this info.
  dum = fread(fid,108,'bit8');
end
for j = 1:ERS.Max_Kanalzahl
% EL_Record_Struktur
  ELS.HEB_Nr = fread(fid,1,'uint16');
  ELS.UEB_Nr = fread(fid,1,'uint16');
  ELS.Bezeichnung = stringread(fid,11);% seems to work...instead of 15.
  ELS.PEG = fread(fid,1,'uchar');% boolean takes one byte of space.
  ELS.Beschriftung_an_HBOX = stringread(fid,11);
  ELS.negativ = fread(fid,1,'uchar');

  ELS.GE_Nr_an_HBOX = fread(fid,1,'uint16');
  ELS.Reihenfolge = fread(fid,1,'uint16');
  ELS.Gruppe = fread(fid,1,'uint16');
  ELS.Zeilenpos_in_UEB = fread(fid,1,'uint16');
  ELS.Spaltenpos_in_UEB = fread(fid,1,'uint16');
  ELS.Mikrovolt_pro_Masseinheit = fread(fid,1,'double');
  ELS.Dimension = stringread(fid,11);
  ELS.Masseinheiten_pro_cm = fread(fid,1,'double');
  ELS.Mikrovolt_bei_Null_Masseinheit = fread(fid,1,'double');
  ELS.Sonderelektrode = fread(fid,1,'uchar');
  ELS.Noch_Reserviert = fread(fid,1,'uchar');
  
  ELS.Word_1_Spez = fread(fid,1,'uint16');
  ELS.Word_2_Spez = fread(fid,1,'uint16');
  ELS.Byte_Spez = fread(fid,1,'uchar');
  % 80 bytes each...
  
  % ELS isn't necessarily a recorded channel. Watch out.
  if ELS.Reihenfolge > 0 & chan_ind<numChans-2
    %ELS belongs to recorded channel.
    chan_ind = chan_ind + 1;
    clab{tec_chan_ind+chan_ind} = debl(ELS.Bezeichnung);
    % some cosmetic corrections:
    if isempty(clab{tec_chan_ind+chan_ind})
      clab{tec_chan_ind+chan_ind} = ['K' num2str(tec_chan_ind)];
      tec_chan_ind = tec_chan_ind+1;
    end
    if strcmp(clab{tec_chan_ind+chan_ind}(1:2),'FP')
      clab{tec_chan_ind+chan_ind}(2) = 'p';
    end
    s = findstr(clab{tec_chan_ind+chan_ind},'Z');
    if ~isempty(s)
      clab{tec_chan_ind+chan_ind}(s) = 'z';
    end
    out_ELS(tec_chan_ind+chan_ind) = ELS;
    
    %%%%%%%%%%%%%%%%
    % let's suppose every channel has its ELS.Reihenfolge.
    % Otherwise, this could help:
    %%%%%%%%%%%%%%%%
    
  elseif  isempty(debl(ELS.Bezeichnung))&tec_chan_ind<2
    %ELS belongs to a technical channel.
    tec_chan_ind = tec_chan_ind+1;
    clab{tec_chan_ind+chan_ind} = ['K' num2str(tec_chan_ind)];
    out_ELS(tec_chan_ind+chan_ind) = ELS;
  else
    %ELS does not belong to a recorded channel. Toss it away.
  end
end


fseek(fid,0,1);
len = (ftell(fid)-PRS.Seekposition_Datenbeginn)/(4*numChans*fs);
fclose(fid);

endian = 'l';
scale = ([-ones(1,length(clab))].^[out_ELS(:).negativ])*ERS.DigitSpannung_in_Mikrovolt;
return

function str = debl(st, num)
% st character array.
if ~exist('num','var')
  num = 1;
end
%deblank:
str = char(max(double(st),32));
%select only token number num:
for i = 1:(num-1)
  [dum,str] = strtok(str);
end
str = strtok(str);
return

function str = stringread(fid,len)
% st character array, len length of characters to read (+1)
% The next position in the file gives the real length of the character array.
a = fread(fid,1);
str = fread(fid,len-1);
str = char(str(1:min(a,length(str))))';
return