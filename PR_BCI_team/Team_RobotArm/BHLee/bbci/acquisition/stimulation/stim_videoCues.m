function stim= stim_videoCues(stim, varargin)
%STIM_VISUALCUES - Presentation of Visual Cues
%
%Synopsis:
% stim_visualCues(STIM, <OPT>)
%
%Arguments:
% STIM
% OPT: struct or property/value list of optional arguments:
% 'breaks': 
% 'msg_vpos': Scalar. Vertical position of message text object. Default: 0.57.
% 'msg_spec': Cell array. Text object specifications for message text object.
%   Default: {'FontSize',0.1, 'FontWeight','bold', 'Color',[.9 0 0]})

% blanker@cs.tu-berlin.de, Jul-2007

global VP_CODE VP_SCREEN DATA_DIR

if ~isstruct(stim),
  error('first argument STIM must be a struct');
end
if ~isfield(stim, 'cue'),
  error('first argument STIM must have a field ''cue''.');
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filename', '', ...
                  'test', 0, ...
                  'position', VP_SCREEN, ...
                  'bv_host', 'localhost', ...
                  'background', 0.5*[1 1 1], ...
                  'countdown', 7, ...
                  'countdown_fontsize', 0.3, ...
                  'duration_intro', 10000, ...
                  'msg_vpos', 0.1, ...
                  'msg_spec', {'FontSize',0.1, 'FontWeight','bold', ...
                               'Color',0.3*[1 1 1]}, ...
                  'breaks',inf, ...
                  'break_minevents',7, ...
                  'break_markers', [249 250], ...
                  'break_msg', 'Short break for %d s', ...
                  'break_countdown', 7, ...
                  'delete_obj', [], ...
                  'test_duration_intro', 1000, ...
                  'test_nEvents', 5, ...
                  'msg_fin','fin');

stim= set_defaults(stim, ...
                   'msg_intro', 'be prepared');

if length(opt.breaks)==1,
  opt.breaks= [opt.breaks 10];
end

if ~isempty(opt.bv_host),
  bvr_checkparport;
end

delete(opt.delete_obj(find(ishandle(opt.delete_obj))));
% if ~isfield(stim.cue, 'handle'),
%   opt.handle_background= stimutil_initFigure(opt);
%   H= num2cell(stimutil_cueStrings(stim.cue.string, opt));
%   [stim.cue.handle]= deal(H{:});
% end

[opt.handle_msg, opt.handle_background]= stimutil_initMsg(opt);
if ~isfield(stim, 'prelude') & ~isfield(stim, 'desc'),  
  %% Fuer ungeduldige (wird später eh angezeigt)
  set(opt.handle_msg, 'String',stim.msg_intro);
end

fprintf('loading videos... ');
opt.movies= cell(1, length(stim.cue));
for i= 1:length(stim.cue),
  opt.movies{i}= aviread([DATA_DIR 'video/' stim.cue(i).bodypart '.avi']);
end
[opt.resMovieY opt.resMovieX bla]= size(opt.movies{1}(1).cdata); 
fprintf('done.\n');

if opt.test,
  opt.duration_intro= opt.test_duration_intro;
  [stim.cue.nEvents]= deal(opt.test_nEvents);
else
  if isfield(stim, 'desc'),
    stimutil_showDescription(stim.desc, opt);
  end
  if ~isempty(opt.filename),
    bvr_startrecording([opt.filename VP_CODE]);
  else
    warning('!*NOT* recording: opt.filename is empty');
  end
end
ppTrigger(251);

if ~isfield(opt, 'handle_cross') | isempty(opt.handle_cross),
  opt.handle_cross= stimutil_fixationCross(opt);
end

if isfield(stim, 'prelude'),
  show_cue_sequence(stim.prelude, opt, 'add_to_marker',10);
  pause(1);
end

show_cue_sequence(stim, opt);
ppTrigger(254);

set(opt.handle_msg, 'String',opt.msg_fin, 'Visible','on');
pause(1);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end

pause(5);

delete(opt.handle_msg);  

return;




function show_cue_sequence(stim, varargin)

stim= set_defaults(stim, ...
                   'msg_intro', 'Be prepared');
if ~isfield(stim.cue, 'marker'),
  [stim.cue.marker]= deal(NaN);
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'add_to_marker', 0);

nClasses= length(stim.cue);
nEvents= sum([stim.cue(:).nEvents]);

cue_sequence= [];
for cc= 1:nClasses,
  if isnan(stim.cue(cc).marker),
    stim.cue(cc).marker= cc;
  end
  if size(stim.cue(cc),1)==1,
    stim.cue(cc).timing(2,:)= zeros(size(stim.cue(cc).timing));
  end
  cue_sequence= cat(1, cue_sequence, cc*ones(stim.cue(cc).nEvents,1));
end
cue_sequence= cue_sequence(randperm(nEvents));

set(opt.handle_msg, 'String',stim.msg_intro, 'Visible','on');
pause(opt.duration_intro/1000);
set(opt.handle_msg, 'Visible','off');

if ~opt.test,
  pause(1)
  stimutil_countdown(opt.countdown, opt);
  ppTrigger(252);
  pause(1);
end

waitForSync;
for ei= 1:nEvents,
  cc= cue_sequence(ei);
  tim= stim.cue(cc).timing(1,:) + rand(1,3).*stim.cue(cc).timing(2,:);
  
  set(opt.handle_cross, 'Visible','on');
  if tim(1)>0,
    ppTrigger(101);
    drawnow;
    waitForSync(tim(1));
  end
  
  tic;
  movie(opt.handle_background, opt.movies{stim.cue(cc).marker}, ...
        [1], 12, ...
        [opt.position(4)/2-opt.resMovieX/2 ...
         opt.position(4)/2-opt.resMovieY/2 0 0]);
  set(opt.handle_cross, 'Visible','off');
%  drawnow;
  ppTrigger(stim.cue(cc).marker + opt.add_to_marker);
  %% This is needed on my laptop:
  set(gcf,'Visi','off'); set(gcf, 'Visi','on');
  %% --
  rc= waitForSync(tim(2));
  if rc>0
    warning(sprintf('Video took %.0f msec too long.', rc));
  end
  
  if tim(3)>0,
    ppTrigger(100);
    drawnow;
    waitForSync(tim(3));
  end
  
  if mod(ei, opt.breaks(1))==0 & ei<nEvents-opt.break_minevents,
    set(opt.handle_cross, 'Visible','off');
    stimutil_break(opt, 'break_duration',opt.breaks(2));
  end
end
ppTrigger(253);
pause(1);
set(opt.handle_cross, 'Visible','off');
