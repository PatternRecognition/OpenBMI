function [BF,BU] = parse_ukbf(file)
% function [BF,BU] = parse_ukbf(file)
%
% This function reads a doc file from UKBF database, 
% converts it to html and then to txt and extracts
% the fields Befund and Beurteilung
%
% Arguments:
%    file: the doc file to be parsed without extention
%
% Returns:
%    BF: a string containing the Befund of the doc file
%    BU: a string containing the Beurteilung of the doc file
%
%

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
  end 
end

fclose(fid);

% for testing
%for i=1:size(str_befund,2)
%  disp([str_befund(i) ' : ' num2str(double(str_befund(i)))]);
%end
%return;

% parse Befund
BF = parse_string(str_befund(3:end));
% parse Beurteilung
BU = parse_string(str_beurteilung(3:end));

% clean up
[s,w] = system(['rm -f ' file '.html ' file '.txt ' file '*.png']);

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
