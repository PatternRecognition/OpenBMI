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
%               .background:   to make available work in the background (slower)
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

% Guido Dornhege, 05/03/04

persistent movie opt file poi fps number

switch type
  case 'open'
    file = varargin{1};
    fps = varargin{2};
    opt = propertylist2struct(varargin{3:end});
    opt = set_defaults(opt, 'overwrite',0);
    if exist([file '.avi'],'file'),
      if opt.overwrite,
        if isunix,
          stat=unix(sprintf('rm -f %s.avi %s_???.avi', file, file));
          if stat~=0,
            keyboard;
          end
        else
          dos(sprintf('del %s.avi %s_???.avi', file, file));
        end
      else
        error('file exist, please remove it first');
      end
    end
    poi = 0;
    
%                       'maxFileSize',1000000000*4,...
    opt = set_defaults(opt,...
                       'fps',fps,...
                       'resolution',150,...
                       'maxFileSize',1000000000*2,...
                       'image_format','png',...
                       'other_image_options',{},...
                       'background',false,...
                       'compression','none',...
                       'quality',100);

    if file(1)~='/' & file(2)~=':',
      global DATA_DIR
      file = [DATA_DIR 'eegVideo/' file];
    end
    
    number = 1;
    movie = avifile([file '.avi'], 'fps',opt.fps, ...
                    'Compression',opt.compression, 'quality',opt.quality);
    
    
 case 'exit'
  
  end_frame(movie,opt);
  if number==1,
    if isunix,
      stat= unix(sprintf('mv %s.avi %s_001.avi', file, file));
      if stat~=0,
        keyboard;
      end
    else
      cmd= sprintf('rename %s.avi %s_001.avi', file, file);
      dos(cmd);
    end
  end    
  if isunix
    [pathstr, filestr]= fileparts(file);
    cmd= sprintf('cd %s; transcode -i %s_%03d.avi -use_rgb -z -y xvid -o tmp_%s_%03d.avi', pathstr, filestr,number,filestr,number);
    unix(cmd);
    stat= unix(sprintf('rm -f %s_%03d.avi', file,number));
  end
  
  if isunix,
    [pathstr, filestr]= fileparts(file);
    if number>1,
      cmd= sprintf('cd %s; avimerge -o %s.avi -i tmp_%s_???.avi', pathstr, filestr, filestr);
      stat= unix(cmd);
      if stat~=0,
        keyboard;
      end
      stat= unix(sprintf('rm -f %s/tmp_%s_???.avi', pathstr, filestr));
      if stat~=0,
        keyboard;
      end
    else
      unix(sprintf('mv %s/tmp_%s.avi %s.avi', pathstr, filestr, file));
    end
  end
 
 case 'frame'
  opt_frame = propertylist2struct(varargin{2:end});
  F = get_frame(varargin{1},opt);
  if isfield(opt_frame, 'fadefactor') & ~isempty(opt_frame.fadefactor),
    F.cdata= uint8( double(F.cdata) * opt_frame.fadefactor );
  end
  [movie, number] = add_frame(movie, F, opt, file, number);
  poi = poi+1;
%  fprintf('\r %5.2f seconds done        ',poi/opt.fps);
end
