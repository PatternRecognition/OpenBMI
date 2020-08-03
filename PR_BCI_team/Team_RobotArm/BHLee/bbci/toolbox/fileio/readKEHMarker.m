function [mrk,classDef]= readKEHMarker(mrkName, fs)
%[mrk,classDef]= readKEHMarker(mrkName, <fs=128>)
%
% IN   mrkName    - name of marker file (no extension),
%                   relative to EEG_RAW_DIR unless beginning with '/'
%      fs         - calculate marker positions for sampling rate fs,
%                   default: 128
%
% OUT  mrk        - struct for event markers
%         .toe    - type of event
%         .pos    - position in data points (for lagged data)
%         .fs     - sampling interval
%      classDef   - cell array, each entry of the first row holds the 
%                   codes (toe) of the corresponding class. A second 
%                   row contains class names.
%
% GLOBZ  EEG_RAW_DIR

global EEG_RAW_DIR

if ~exist('fs', 'var'), fs=128; end
if ~exist('markerTypes', 'var'), markerTypes={'Stimulus','Response'}; end
if ~exist('flag', 'var'), flag=[1 -1]; end

if mrkName(1)==filesep,
  fullName= mrkName;
else
  fullName= [EEG_RAW_DIR mrkName];
end

[dum,dum,mrk_fs,dum,dum,dum,dum] = readKEHHeader(fullName);
lag= mrk_fs/fs;


fid= fopen([fullName '.marke'], 'r'); 
if fid==-1, error(sprintf('%s.marke not found', fullName)); end

fseek(fid,0,1);% skip to the end
endpos = ftell(fid);
fseek(fid,0,-1);% skip to start

mrk.pos= [];
mrk.toe= [];
ei= 0;
MRS = struct('Markenquelle','');

classDef = {};
while ftell(fid)~=endpos,
  ei= ei+1;
  MRS.Markenquelle = fread(fid,1,'uint16');
  MRS.aktiv = fread(fid,1,'uchar');
  MRS.Technik_Wert_1 = fread(fid,1,'int32');% this is the pressed key.
  MRS.Technik_Wert_2 = fread(fid,1,'int32');
  a = fread(fid,1,'uchar');
  MRS.Code_Char = char(a);
  MRS.Code_Word = fread(fid,1,'uint16');
  MRS.Anzeige_String = stringread(fid,80);%char(fread(fid,80))';
  MRS.Kanal_Nr_Marke = fread(fid,1,'uint16');
  MRS.EL_Bez_Marke = stringread(fid,16);
  MRS.EL_Nr_HBOX_Marke = fread(fid,1,'uint16');
  MRS.Marken_Typ_Nummer = fread(fid,1,'uint16');
  MRS.Messwert_im_Kanal = fread(fid,1,'int32');
  MRS.automatisch_gesetzt = fread(fid,1,'uchar');
  MRS.Marke_wurde_editiert = fread(fid,1,'uchar');
  MRS.Kommentar_String = stringread(fid,22);
  MRS.Record_ID = stringread(fid,9);
  MRS.DVideo_Time_Code = stringread(fid,17);
  MRS.Datenpunkt = fread(fid,1,'uint32');
  % from beginning of the file or from Seekposition?...
  MRS.Video_Frame = fread(fid,1,'uint32');
  
  mrk.toe(ei)=  MRS.Technik_Wert_1;
  mrk.pos(ei)= ceil(MRS.Datenpunkt/lag);% this is from beginning of file.
  if ~any([classDef{1,:}]==MRS.Technik_Wert_1)
    classDef{1,end+1} = MRS.Technik_Wert_1;
    ind = find(double(MRS.Anzeige_String)>31);
    classDef{2,end} = MRS.Anzeige_String(ind);
  end
end

mrk.fs= fs;
% Markers might be in the wrong order!
mrk = mrk_sortChronologically(mrk);

fclose(fid);
return

function str = stringread(fid,len)
% st character array, len length of characters to read (+1)
% The next position in the file gives the real length of the character array.
a = fread(fid,1);
str = fread(fid,len-1);
str = char(str(1:min(a,length(str))))';
return
