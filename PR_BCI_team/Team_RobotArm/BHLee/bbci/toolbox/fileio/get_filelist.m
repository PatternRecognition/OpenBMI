function file= get_filelist(spec, varargin)

%% TODO: should also handle: spec is cell array

global EEG_RAW_DIR EEG_MAT_DIR

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'ext', 'eeg', ...
                 'folder', EEG_RAW_DIR, ...
                 'require_match', 0);

if isdefault.ext,
  is= find(spec=='.', 1, 'last');
  if ~isempty(is) && is<length(spec) && is>=length(spec)-4,
    ext= spec(is+1:end);
    if ~ismember('*',ext) && ~ismember('/',ext),
      opt.ext= ext;
    end
  else
    spec= [spec '.' opt.ext];
  end
else
  spec= [spec '.' opt.ext];
end

if isdefault.folder,
  switch(lower(opt.ext)),
   case {'eeg','vhdr','vmrk'},
    opt.folder= EEG_RAW_DIR;
   case 'mat',
    opt.folder= EEG_MAT_DIR;
  end
end

[filepath, dmy]= fileparts(spec);
if ~isabsolutepath(spec),
  spec= [opt.folder filesep spec];
end

dd= dir(spec);
if isempty(dd),
  if opt.require_match,
    msg= sprintf('no files matching ''%s'' found', spec);
    error(msg);
  end
  file= {};
  return;
end

if length(dd)==1,
  file= strcat(filepath, filesep, dd.name);
else
  file= strcat(filepath, filesep, {dd.name});
end
if ~isdefault.ext,
  file= strrep(file, ['.' opt.ext], '');
end
