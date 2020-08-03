function replay(name, number, varargin)
%REPLAY - generate replays of logged BBCI feedbacks
%
%Synopsis:
% replay(NAME, NUMBER, <OPT>)
%
%Arguments
% NAME: the name of a feedback, e.g. 'brainpong'
% NUMBER: number of the generated logfile. 
% OPT - struct or property/value list of optional properties:
%   .start: time to start replay [s], default 0
%   .stop: time to stop replay [s], default inf
%   .save: file name to save replay, [] for not saving (default)
%   .speedup: number between 0 and 1: 0=real time, 1:fast scan
%   .opt_movie: optinal properties for generating the movie
%   .opt_fb: properties to override original feedback properties
%
% SEE ALSO: load_log, make_moviefile, do_set

% Author(s): Matthias Krauledat 04/03/04, Benjamin Blankertz Feb-2006

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filename', name, ...
                  'start', 0, ...
                  'stop', inf, ...
                  'save', [], ...
                  'speedup',0, ...
                  'position', [], ...
                  'skip_long_pauses', 1, ...
                  'opt_movie', '', ...
                  'opt_fb', '', ...
                  'force_set', {});

[fb_opt,dum,init_file]= load_log(opt.filename, number);
fb_opt = set_defaults(fb_opt,...
                      'fs',25,...
                      'repetitions',1);
fb_opt= set_defaults(opt.opt_fb, fb_opt);
fb_opt.log= 0;

% init-case for the feedback.
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

if ~isempty(init_file)
  d = cd;
  c = find(init_file =='/');
  c = c(end);
  cd(init_file(1:c));
  handle = feval(init_file(c+1:end),fig,fb_opt);
  cd(d);
else
  handle = feval(['feedback_' name '_init'], fig, fb_opt);
end

if isstruct(handle)
    handle= fb_handleStruct2Vector(handle);
end
do_set('init', handle, name, fb_opt);
waitForSync(0);
if ~isempty(opt.save),
  make_moviefile('open',opt.save, fb_opt.fs, opt.opt_movie);
end
out= load_log;

count= floor(opt.start/1000*fb_opt.fs);
while ~isempty(out)
  if isstruct(out)
    %fb_opt has changed. ???
  end
  if iscell(out),
    if strcmp(out{1}, 'BLOCKTIME'),
      out= load_log;
      continue;
    end
    frameno= out{2};
    msec= frameno*1000/fb_opt.fs;
    fprintf('\r%010.3f ', msec/1000);
    if msec > opt.stop*1000,
      break;
    end
    if msec <= opt.start*1000,
      do_set(out{4}, out{5:end});
    else
      % optionally skip pauses longer than 1s
      if opt.skip_long_pauses & (frameno-count > fb_opt.fs),
        count= frameno-1;
      end
      while frameno>count+1,
        % some frames must be drawn with the same setting.
        count= count+1;
        do_set('+');
        if opt.save,
          % STORE FRAME
          make_moviefile('frame', fig);
        else
          waitForSync(1000/fb_opt.fs*(1-opt.speedup));
        end
      end
      if frameno>count, 
        % Next frame needs to be drawn.
        count=count+1;
        do_set('+');
        if opt.save,
          make_moviefile('frame', fig);
        else
          waitForSync(1000/fb_opt.fs*(1-opt.speedup));
        end
      end
      do_set(out{4}, out{5:end});
      if ~isempty(opt.force_set),
        for kk= 1:length(opt.force_set),
          do_set(opt.force_set{kk}{1}, opt.force_set{kk}{2:end});
        end
      end
      
    end
  end
  out = load_log;
end
fprintf('\n');

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

load_log('exit');
