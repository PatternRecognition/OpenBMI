function replay_bb(name, number, varargin)
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
                  'start', 0, ...
                  'stop', inf, ...
                  'save', [], ...
                  'speedup',0, ...
                  'position', [], ...
                  'skip_long_pauses', 1, ...
                  'opt_movie', '', ...
                  'opt_fb', '');

fb_opt= load_log(name, number);
fb_opt = set_defaults(fb_opt,...
                      'fs',25,...
                      'repetitions',1);
fb_opt= set_defaults(opt.opt_fb, fb_opt);
fb_opt.log= 0;

% init-case for the feedback.
fig= figure;
if ~isempty(opt.position),
  set(fig, 'Position',opt.position);
end
HH= feval(['feedback_' name '_init'], fig, fb_opt);
handle= fb_handleStruct2Vector(HH);
do_set('init', handle, name, fb_opt);
waitForSync(0);
if ~isempty(opt.save)
  make_moviefile('open',opt.save, fb_opt.fs, opt.opt_movie);
end
out= load_log;

count= opt.start*fb_opt.fs;
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
    msec= frameno*fb_opt.fs;
    fprintf('\r%010.4f ', msec/1000);
    if msec > opt.stop*1000,
      break;
    end
    if msec < opt.start*1000,
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
      do_set(out{4}, out{5:end});
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
    end
  end
  out = load_log;
end
fprintf('\n');

if opt.save,
  if ~isempty(opt.opt_movie) & isfield(opt.opt_movie,'stoptime'),
    for i = 1:opt.opt_movie.stoptime*fb_opt.fs/1000;
      make_moviefile('frame', fig);
    end
  end
  make_moviefile('exit');
end

load_log('exit');
