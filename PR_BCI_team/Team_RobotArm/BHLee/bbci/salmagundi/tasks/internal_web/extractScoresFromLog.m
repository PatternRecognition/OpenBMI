function score= extractScoresFromLog(log_file, upper_bound)

if nargin<2,
  upper_bound= inf;
end

score= [];
fid= fopen(log_file);
while ~feof(fid),
  str= fgetl(fid);
  ii= strfind(str, 'bit/min');
  if isempty(ii), 
    ii= strfind(str, 'bits/min');
  end
  if isempty(ii), continue; end
  is= min(find(str==''''));
  ie= is + min(find(str(is+1:end)==' '));
  bitrate= str2num(str(is+1:ie-1));
  score= cat(2, score, bitrate);
end
fclose(fid);

valid= find(~isinf(score) & ~isnan(score) & score<upper_bound);
score= score(valid);
