function make_movie(file, varargin)
%REPLAY - generate replays of logged BBCI feedbacks
%
%Synopsis:
% make_movie(file, <OPT>)
%
%Arguments
% file - file to load
% OPT - struct or property/value list of optional properties:
%   .start: time to start replay [s], default 0
%   .stop: time to stop replay [s], default inf
%   .save: file name to save replay, [] for not saving (default)
%   .speedup: number between 0 and 1: 0=real time, 1:fast scan
%   .opt_movie: optinal properties for generating the movie
%   .opt_fb: properties to override original feedback properties
%
% SEE ALSO: load_log, make_moviefile, do_set

% Author(s): Guido 04/04/06

global EEG_VIDEO_DIR

opt= propertylist2struct(varargin{:});

opt= set_defaults(opt, ...
                  'start', 0, ...
                  'stop', inf, ...
                  'save', [], ...
                  'speedup',0, ...
                  'position', [], ...
                  'opt_movie', '', ...
                  'opt_fb', '');

if ~isempty(opt.save)
  if (isunix & opt.save(1)~='/') | (~isunix & opt.save(2)~=':')
    opt.save = [EEG_VIDEO_DIR opt.save];
  end
  c = strfind(opt.save,filesep);
  mkdir_rec(opt.save(1:c(end)));
end


feedback = eegfile_loadMatlab(file,{'feedback'});

global EEG_RAW_DIR interrupt_movie

if (isunix & file(1)~='/') | (~isunix & file(2)~=':')
  file = [EEG_RAW_DIR file];
end

c = strfind(file,filesep);
c = c(end);
opt_file = [file(1:c) 'log/' feedback.file{1}];

load(opt_file);

fb_opt= set_defaults(opt.opt_fb, fb_opt);
fb_opt.log = 0;

close(gcf);
fig= figure;

if ~isempty(opt.position),
  set(fig, 'Position',opt.position);
  drawnow;
  set(fig, 'Position',opt.position);  %% hack due to fluxbox
  drawnow;                            %% bug
%  pos= get(fig, 'Position');
%  if ~isequal(pos, opt.position),
%    fprintf('set(fig, ''Position'',opt.position); dbcont\n');
%    keyboard
%  end
end

if isfield(feedback,'init_file') & exist(feedback.init_file)
  d = cd;
  c = find(feedback.init_file ==filesep);
  c = c(end);
  cd(feedback.init_file(1:c));
  filename = feedback.init_file(c+1:end);
  if strcmp(filename(end-1:end),'.m')
    filename = filename(1:end-2);
  end
  handle = feval(filename,fig,fb_opt);
  cd(d);
else
  handle = feval([fb_opt.type '_init'], fig, fb_opt);
end

if ~isempty(opt.position),
  set(fig, 'Position',opt.position);
  drawnow;
  set(fig, 'Position',opt.position);  %% hack due to fluxbox
  drawnow;                            %% bug
%  pos= get(fig, 'Position');
%  if ~isequal(pos, opt.position),
%    fprintf('set(fig, ''Position'',opt.position); dbcont\n');
%    keyboard
%  end
end

if isstruct(handle)
    handle= fb_handleStruct2Vector(handle);
end


set(fig,'KeyPressFcn','global interrupt_movie; interrupt_movie = 1;');
interrupt_movie = 0;

for i = 1:size(feedback.initial,1)
  for j = 1:length(feedback.initial{i,1})
    set(handle(i),feedback.initial{i,1}{j},feedback.initial{i,2}{j});
  end
end

waitForSync(0);
if ~isempty(opt.save),
  make_moviefile('open',opt.save, fb_opt.fs, opt.opt_movie);
end


pos_old = 0;
start= floor(opt.start*feedback.fs);
ende= floor(opt.stop*feedback.fs);

for out = 1:length(feedback.update.pos)
  
  pos = feedback.update.pos(out);
  
  if pos_old>ende | interrupt_movie
    break;
  end
  
  if pos>pos_old
    if pos>=start  
      if opt.save,
        while pos-pos_old>0 & ~interrupt_movie
          make_moviefile('frame', fig);
          pos_old = pos_old+(feedback.fs/fb_opt.fs);
        end
      else
        drawnow;
        waitForSync(((pos-pos_old)*1000/feedback.fs)*(1-opt.speedup));
        pos_old = pos;
      end
    else
      pos_old = pos;
    end
    
  end
  
  if  interrupt_movie
    break;
  end

  for j = 1:length(feedback.update.prop{out})
    set(handle(feedback.update.object(out)),feedback.update.prop{out}{j},feedback.update.prop_value{out}{j});
  end
  
end


if opt.save,
  if ~isempty(opt.opt_movie) & isfield(opt.opt_movie,'freeze_out'),
    for i = 1:opt.opt_movie.freeze_out*fb_opt.fs;
      make_moviefile('frame', fig);
    end
  end
  if ~isempty(opt.opt_movie) & isfield(opt.opt_movie,'fade_out'),
    Ni= opt.opt_movie.fade_out*fb_opt.fs;
    for i = 1:Ni;
      make_moviefile('frame', fig, 'fadefactor',(Ni-i)/Ni);
    end
  end
  make_moviefile('exit');
end

close all
