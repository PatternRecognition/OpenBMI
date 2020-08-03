function mov = make_moviefile(file,fs,varargin);

opt = propertylist2struct(varargin{:});


if isunix
  global LOG_DIR
  if isempty(LOG_DIR)
    LOG_DIR = '/tmp/';
  end
  opt = set_defaults(opt,...
                     'fps',fs,...
                     'resolution',150,...
                     'image_format','png',...
                     'other_image_options',{{}},...
                     'max_size',1000,...
                     'transcode_options',{{}});
  
  
else
  opt = set_defaults(opt,...
                     'fps',fs,...
                     'compression','Cinepak',...
                     'quality',100);
  
  global LOG_DIR
  if isempty(LOG_DIR)
    LOG_DIR = 'g:\eeg_temp\log\';
  end
  
  if file(2)~=':'
    file = [LOG_DIR file];
  end
  mov = avifile(file,'fps',opt.fs,'Compression',opt.compression,'quality',opt.quality);
end


