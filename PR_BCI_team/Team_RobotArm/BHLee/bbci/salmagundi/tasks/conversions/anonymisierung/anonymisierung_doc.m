function info = anonymisierung_doc(doc,output,ignore_portion,quality);
%
% usage: info = anonymisierung_doc(doc,output,ignore_portion,quality);
%
% input: doc: the document to process
%        output: name of the jpg-file (default '', no image processing)
%        ignore_portion: portion of the top to ignore (default 0.22)
%        quality: quality of the jpg (dpi) (defualt 300)
%
% output: info: the information about the doc file as struct

if nargin<4
  quality = 300;
end

if nargin<3
  ignore_portion = 0.22;
end

if nargin<2
  output = '';
end

% do antiword to extract text

[a,b]= system(['antiword ' doc]);

if a~=0
  error('cannot read document');
end

text = b;

% extract relevant part
c = strfind(text,'EEG-Befund');

text = text(c(1)+length('EEG-Befund')+1:end);

while text(1)==10
  text(1)='';
end
 

c = strfind(text,'[pic]');

text = text(1:c(1)-1);

while text(end)==10
  text(end)='';
end
 
% extract picture if desired by antiword and convert. Ignore relevant picture part
if ~isempty(output)
  [a,b] = system(['cp ' doc ' /tmp/tmp_doc.doc']);
  d = cd;
  cd('/tmp/');
  [a,b] = system(['antiword -p a4 -i 0 tmp_doc.doc >tmp_doc.ps']);
  [a,b] = system(['convert -density ' int2str(quality) 'x tmp_doc.ps tmp_doc.jpg']);
  
  cd(d);
  
  dd = dir('/tmp/tmp_doc.jpg*');
  if length(dd)>1
    arr = zeros(1,length(dd));
    for i = 1:length(dd)
      arr(i) = str2num(dd(i).name(length('tmp_doc.jpg.')+1:end));
    end
    [arr,ind] = max(arr);
    system(['mv /tmp/' dd(ind).name ' /tmp/tmp_doc.jpg']);
    system(['rm -f /tmp/tmp_doc.jpg.*']);
  end
  system(['rm -f /tmp/tmp_doc.doc']);
  system(['rm -f /tmp/tmp_doc.ps']);
  
  bild = imread('/tmp/tmp_doc.jpg');
  system(['rm -f /tmp/tmp_doc.jpg']);
  
  bb = all(bild==255,3);
  bild(all(bb,2),:,:) = [];
  bild(:,all(bb,1),:) = [];
  bild(1:round(size(bild,1)*ignore_portion),:,:) = [];
  
  imwrite(bild,output,'JPG');
end

% extract relevant informations from text + anonymization
info2 = extract_table(text);

info = struct('Anforderung',get_value(info2,'Anforderung'));
info.geschlecht= get_value(info2,'Patient');
c = strfind(info.geschlecht,'(');
if isempty(c)
  info.geschlecht = '';
else
  info.geschlecht = info.geschlecht(c(end)+1);
end
geburtsjahr= get_value(info2,'geb');
c = strfind(geburtsjahr,'.');
if isempty(c)
  info.geburtsjahr = [];
else
  info.geburtsjahr = str2num(geburtsjahr(c(end)+1:end));
end
info.anamnese = get_value(info2,'Anamnese');
info.neurologischer_befund = get_value(info2,'Neurologischer Befund');
info.medikation = get_value(info2,'Medikation');
info.diagnose = get_value(info2,'Diagnose');
info.fragestellung = get_value(info2,'Fragestellung');
info.ableitung = get_value(info2,'Ableitung');
info.mta = get_value(info2,'MTA');
info.verhalten = get_value(info2,'Verhalten');
info.artefakte = get_value(info2,'Artefakte');
info.fachrichtung = get_value(info2,'Fachrichtung');
info.station = get_value(info2,'Station');
info.anforderer = get_value(info2,'Anforderer');
info.eegnr = get_value(info2,'EEG-Nr.');
if strcmp(info.eegnr(1:4),'Nr. ');
  info.eegnr(1:4)  = '';
end
info.hv = get_value(info2,'HV');
info.bewusstsein = get_value(info2,'Bewusstsein');
info.kommentar = get_value(info2,'Kommentar');

if length(geburtsjahr)==length('xx.xx.xxxx') & geburtsjahr(3)=='.' & geburtsjahr(6)=='.' & ~isempty(str2num(geburtsjahr([1:2,4:5,7:10])))
  info.alter = floor(str2num(geburtsjahr(7:10))-str2num(info.ableitung(7:10))+ (str2num(geburtsjahr(4:5))-str2num(info.ableitung(4:5)))/12 +(str2num(geburtsjahr(4:5))-str2num(info.ableitung(4:5)))/12/30); 
else
  info.alter = str2num(info.ableitung(length('xx.xx.')+(1:4)))-info.geburtsjahr;
end

c = strfind(text,sprintf('\nBefund:'));
info.befund = '';
cc = strfind(text(c(1)+1:end),sprintf('\n'));
cc = cc(1)+c(1);

c = cc;
cc = strfind(text(c+1:end),sprintf('\n'));
cc = cc(1)+c;

while cc-c>2
  info.befund = [info.befund,' ',text(c+1:cc-1)];
  c = cc;
  cc = strfind(text(c+1:end),sprintf('\n'));
  cc = cc(1)+c;
end

c = strfind(text,sprintf('\nBeurteilung:'));
info.beurteilung = '';
cc = strfind(text(c(1)+1:end),sprintf('\n'));
cc = cc(1)+c(1);

c = cc;
cc = strfind(text(c+1:end),sprintf('\n'));
cc = cc(1)+c;

while cc-c>2
  info.beurteilung = [info.beurteilung,' ',text(c+1:cc-1)];
  c = cc;
  cc = strfind(text(c+1:end),sprintf('\n'));
  cc = cc(1)+c;
end

while text(end)==10
  text(end)=='';
end

c = strfind(text,sprintf('\nBefunder:'));
if isempty(c)
  cc = strfind(text,sprintf('\n'));  
  info.befunder = text(cc(end):end);
else
  cc = strfind(text(c(1)+1:end),sprintf('\n'));
  if isempty(cc)
    info.befunder = '';
  else
    info.befunder = text(c(1)+cc(1)+1:end);
  end
end

while length(info.befunder)>0 & info.befunder(1)==' '
  info.befunder(1) = '';
end
while length(info.befunder)>0 & info.befunder(end)==' '
  info.befunder(end) = '';
end

while length(info.beurteilung)>0 & info.beurteilung(1)==' '
  info.beurteilung(1) = '';
end
while length(info.beurteilung)>0 & info.beurteilung(end)==' '
  info.beurteilung(end) = '';
end

while length(info.befund)>0 & info.befund(1)==' '
  info.befund(1) = '';
end
while length(info.befund)>0 & info.befund(end)==' '
  info.befund(end) = '';
end

% inputs made by markus
%   extract Befund and Beurteilung from doc file
[befund,beurteilung] = extract_befund_etc(doc(1:end-4));
info.befund = befund;
info.beurteilung = beurteilung;
% inputs made by markus end here


% helper functions

function val = get_value(info,string);

val = strmatch(string,info(:,1));
if length(val)<1
  return;
end
if length(val)>1
  error('not able to extract information');
end
val = info{val,2};
if isempty(val)
  val = '';
end


function table = extract_table(text);

c = [0,find(text==10),length(text)];
te = cell(1,length(c)-1);
for i = 1:length(te)
  te{i} = text(c(i)+1:c(i+1)-1);
end

tab = {};
stat = 0;
for i = 1:length(te)
  if length(te{i})>0 & te{i}(1)==124;
    if stat
      tab{end} = {tab{end}{:},te{i}};
    else
      tab{end+1} = {te{i}};
      stat = 1;
    end
  else
    stat = 0;
  end
end

for i = 1:length(tab)
  for j = 1:length(tab{i})
    te = tab{i}{j};
    c = find(te==124);
    tab{i}{j} = cell(1,length(c)-1);
    for k = 1:length(c)-1
      tab{i}{j}{k} = te(c(k)+1:c(k+1)-1);
      while length(tab{i}{j}{k})>1 & tab{i}{j}{k}(1)==32 & tab{i}{j}{k}(2)==32
        tab{i}{j}{k}(1)='';
      end
      while length(tab{i}{j}{k})>1 & tab{i}{j}{k}(end)==32 & tab{i}{j}{k}(end-1)==32
        tab{i}{j}{k}(end)='';
      end
    end
  end
end

i = 1;
while i<=length(tab)
  if length(tab{i}{1})>2
    tab2 = cell(1,length(tab{i}));
    for j = 1:length(tab2)
      tab2{j} = tab{i}{j}(3:end);
      tab{i}{j} = tab{i}{j}(1:2);
    end
    tab{end+1} = tab2;
  end
  i = i+1;
end

for i = 1:length(tab)
  j = 2;
  while j<=length(tab{i})
    if isempty(tab{i}{j}{1}) | all(tab{i}{j}{1}==32) | (tab{i}{j-1}{1}(end)~=':' & tab{i}{j-1}{1}(end)~=' ')
      tab{i}{j-1}{1} = [tab{i}{j-1}{1} tab{i}{j}{1}];
      tab{i}{j-1}{2} = [tab{i}{j-1}{2} tab{i}{j}{2}];
      tab{i} = {tab{i}{1:j-1},tab{i}{j+1:end}};
    end
    j = j+1;
  end
end

table = {};
for i = 1:length(tab)
  for j = 1:length(tab{i})
    for k = 1:2
       while length(tab{i}{j}{k})>0 & tab{i}{j}{k}(1)==32
         tab{i}{j}{k}(1)='';
       end
       while length(tab{i}{j}{k})>0 & tab{i}{j}{k}(end)==32
         tab{i}{j}{k}(end)='';
       end
    end
    if ~isempty(tab{i}{j}{1})
      table = [table;tab{i}{j}];
    end
  end
end


% functions written by markus

% This function reads a doc file from UKBF database, 
% converts it to html and then to txt and extracts
% the fields Befund and Beurteilung
function [Befund,Beurteilung] = extract_befund_etc(file)
% function [Befund,Beurteilung] = extract_befund_etc(file)
%
% Arguments:
%    file: the doc file to be parsed without extention
%
% Returns:
%    Befund: a string containing the Befund of the doc file
%    Beurteilung: a string containing the Beurteilung of the doc file
%
% Comment: this function also extracts the Befunder of an EEG, but 
%          this field ist not set in all documents, so this feature
%          is not used in the callung function (see above)


% convert the doc file to html
[s,w] = system(['wvWare -cutf-8 ' file '.doc > ' file '.html']);
if s~=0
  error('cannot read document');
end

% convert the html file to text
[s,w] = system(['html2text ' file '.html > ' file '.txt']);
if s~=0
  error('cannot read document');
end

% start parsing
fid=fopen([file '.txt']);

id_befund = [66 8 66 101 8 101 102 8 102 117 8 117 110 8 110 100 8 100 58 8 58];
id_beurteilung = [66 8 66 101 8 101 117 8 117 114 8 114 116 8 116 101 8 101 105 8 105 108 8 108 117 8 117 110 8 110 103 8 103 58 8 58];
id_befunder = [66 8 66 101 8 101 102 8 102 117 8 117 110 8 110 100 8 100 101 8 101 114 8 114 58 8 58];

str_befund = '';
str_beurteilung = '';
str_befunder = '';

code = 0;

% extract befund and beurteilung

while 1
  tline = fgetl(fid);

  if ~ischar(tline)
    break
  else
    dline = double(tline);
    if size(dline) == size(id_befund)
      if dline == id_befund
        code = 1;
	tline = '';
      end
    elseif size(dline) == size(id_beurteilung)
      if dline == id_beurteilung
        code = 2;
	tline = '';
      end
    elseif size(dline) == size(id_befunder)
      if dline == id_befunder
	code = 3;
	tline = '';
      end
    end
  end  
  
  switch code
   case 1
    str_befund = [str_befund ' ' tline];
   case 2
    str_beurteilung = [str_beurteilung ' ' tline];
   case 3
    if size(tline,2)>1
      str_befunder = [str_befunder ' ' tline];
    end
  end 
end

fclose(fid);

% parse Befund
Befund = parse_string(str_befund(3:end));
% parse Beurteilung
Beurteilung = parse_string(str_beurteilung(3:end));
% Befunder
Befunder = str_befunder;

% clean up
[s,w] = system(['rm -f ' file '.html ' file '.txt ']);
[pathstr,name] = fileparts(file);
png_tmp = [cd '/' name '0.png'];
[s,w] = system(['rm -f ' png_tmp]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Snew = parse_string(S)

% Replacements
R = {
    [206 177]     'Alpha';
    [225 129 161] 'Alpha';
    [206 178]     'Beta';
    [225 129 162] 'Beta';
    [206 179]     'Gamma';
    [207 145]     'Theta';
    [225 129 177] 'Theta';
    [206 180]     'Delta';
    [225 129 164] 'Delta';
    [195 164]     'ae';
    [195 132]     'Ae';
    [195 182]     'oe';
    [195 150]     'Oe';
    [195 188]     'ue';
    [195 156]     'Ue';
     };

Snew = S;

for r=1:size(R,1)
  S = Snew;
  Snew = '';
  RD = R{r,1};
  rds = size(RD,2);
  RS = R{r,2};
  rss = size(RS,2);
  count = 0;
  for i=1:size(S,2)-rds
    if count > 0
      count = count - 1;
    else  
      if double(S(i:i+rds-1)) == RD
        Snew = [Snew RS];
	count = rds -1;
      else
        Snew = [Snew S(i)];
      end
    end
  end
  Snew = [Snew S(end-rds+1:end)];
end
