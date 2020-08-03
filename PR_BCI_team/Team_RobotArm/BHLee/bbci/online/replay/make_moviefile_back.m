function make_moviefile(type,varargin);
% MAKE_MOVIEFILE makes a moviefile
%
% usage:
%  <open>    make_moviefile('open',file,fps,proplist);
%  <exit>    make_moviefile('exit');
%  <frame>   make_moviefile('frame',fig);
%
% input:
%  file     - a file name
%  fps      - the sampling rate
%  proplist - a list of properties:
%               .background:   to make available work in the bachground (slower)
%                .resolution (UNIX): the resolution of the movie
%                .image_format (UNIX): the used image format
%                .other_image_options (UNIX): other options for the image
%                .max_size (UNIX): max size of frames for one part of movie
%                  NOTE: With UNIX this programm will make short movies and concatenate them, since the high amount of space.
%                .colormap (UNIX): a colormap
%                .mpgwrite_options (UNIX): other options for mpgwrite
%                .transcode_options (UNIX): options for transcode
%                .compression (WINDOWS) : the name of the compression  
%                .quality (WINDOWS): the quality of the movie
%                
%                the defaults are 
%                       'resolution',150,...
%                       'image_format','png',...
%                       'other_image_options',{},...
%                       'max_size',20,...
%                       'background',true,...
%                       'colormap',[],...
%                       'mpgwrite_options',[],...
%                       'transcode_options',{});
%                       'compression','Cinepak',...
%                       'quality',100,...
%
%
%  fig         -  a pointer on a figure
%
% Guido Dornhege, 05/03/04

global movie
persistent opt file poi fps number

switch type
  case 'open'
    file = varargin{1};
    if exist([file '.avi'],'file')
      error('file exist, please remove it first');
    end
    fps = varargin{2};
    opt = propertylist2struct(varargin{3:end});
    poi = 0;
    
    if isunix
      global LOG_DIR
      if isempty(LOG_DIR)
        LOG_DIR = '/tmp/';
      end
      if file(1)~='/'
        file = [LOG_DIR file];
      end
      opt = set_defaults(opt,...
        'fps',fps,...
        'resolution',150,...
        'image_format','png',...
        'other_image_options',{},...
        'max_size',20,...
        'background',false,...
        'colormap',[],...
        'mpgwrite_options',[],...
        'transcode_options',{});
      
      
      
      movie.fields = struct('cdata',cell(1,opt.max_size),'colormap',cell(1,opt.max_size));;
      movie.pointer = 0;
      movie.file = file;
      % $$$     fi = find_file(file,'avi');
      % $$$     movie.movie = avifile(fi,'fps',opt.fps,'Compression','none');
      % $$$     movie.pointer = 0;
      % $$$     movie.file = file;
      % $$$     movie.filewrite = fi;
    else
      opt = set_defaults(opt,...
        'fps',fps,...
        'resolution',150,...
        'image_format','png',...
        'other_image_options',{},...
        'background',false,...
        'maxFileSize',1000000000*4,...
        'compression','none',...
        'quality',100);
      
      global LOG_DIR
      if isempty(LOG_DIR)
        LOG_DIR = 'i:\eeg_temp\log\';
      end
      
      if file(2)~=':'
        file = [LOG_DIR file];
      end
      if ~isempty(opt.maxFileSize) & opt.maxFileSize<inf
        number = 1;
      else 
        number = [];
      end
      if isempty(number)
        movie = avifile([file '.avi'],'fps',opt.fps,'Compression',opt.compression,'quality',opt.quality);
      else
        movie = avifile(sprintf('%s_%03d.avi',file,number),'fps',opt.fps,'Compression',opt.compression,'quality',opt.quality);
      end  
    end
    
    
  case 'exit'
    end_frame(movie,opt);
  case 'frame'
    if ~isempty(number)
      sz = dir(sprintf('%s_%03d.avi',file,number));
      sz = sz.bytes;
      if sz>opt.maxFileSize
        end_frame(movie,opt);
        number = number+1;
        movie = avifile(sprintf('%s_%03d.avi',file,number),'fps',opt.fps,'Compression',opt.compression,'quality',opt.quality);
      end
    end
      
    F = get_frame(varargin{1},opt);
    movie = add_frame(movie,F,opt);
    poi = poi+1;
    fprintf('\r %5.2f seconds done        ',poi/opt.fps);
end
