% Modifies the TORCS settings file
% assumes that filename and soundfile are given

filename = [TORCS_DIR '\' configFolder '\' SETTINGS_FILE];

if ~exist('soundfile','var') || isempty(soundfile) || strcmp(soundfile,'')
  soundfile = '';
end

% Read in whole file
fid = fopen(filename,'r+');
ll = cell(1,1);
ii=1;
while ~feof(fid)
  l = fgetl(fid);
  if ~isempty(strfind(l,'PROBAND'))
    l = ['PROBAND ' VP_CODE];
  elseif ~isempty(strfind(l,'SOUND_FILE'))
    l = ['SOUND_FILE ' soundfile];
  end
  ll{ii} = l;
  ii=ii+1;
end
fclose(fid);

% Write modified file
fid = fopen(filename,'w');
for ii=1:numel(ll)
  fprintf(fid,'%s\r\n',ll{ii});
end
fclose(fid);