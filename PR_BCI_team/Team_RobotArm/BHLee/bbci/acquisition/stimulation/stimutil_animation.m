function Hout= stimutil_animation(dur, varargin)

persistent opt H state

if isequal(dur, 'init'),
  opt= propertylist2struct(varargin{:});
  opt= set_defaults(opt, ...
                    'axissize', [0.1 0.1], ...
                    'edges', 4, ...
                    'steps_per_transition', 40, ...
                    'fps', 25, ...
                    'color', cmap_rainbow(100, 'val',0.5), ...
                    'color_changing_speed', 0.2);
  opt.positions= [0 0.5 1 0 1 0 0.5 1; 0 0 0 0.5 0.5 1 1 1];
  state.nextpos= randsample(1:size(opt.positions,2), opt.edges);
  state.step=opt.steps_per_transition;
  state.colptr= 0;
  axpos= [[0.5 0.5]-opt.axissize/2, opt.axissize];
  H.axis= axes('position', axpos);
  set(H.axis, 'Visible','off')
  set(H.axis, 'XLim',[0 1], 'YLim',[0 1])
  H.patch= patch(opt.positions(1,state.nextpos), ...
                 opt.positions(2,state.nextpos), ...
                 opt.color(1,:));
  set(H.patch, 'Visible','off');
  if nargout>0,
    Hout= H;
  end
  return;
end

if isequal(dur, 'close'),
  if ~isempty(H),
    delete(H.axis);
  end
  clear opt H
  return;
end

start_time= clock;
set(H.patch, 'Visible','on');
while etime(clock, start_time) < dur,
  if state.step==opt.steps_per_transition,
    state.lastpos= state.nextpos;
    if any(state.lastpos==state.nextpos) || ...
          sort(state.lastpos)==(sort(state.nextpos)),
      state.nextpos= randsample(1:size(opt.positions,2), opt.edges);
    end
    state.step= 0;
  end
  alpha= state.step/opt.steps_per_transition;
  xy= opt.positions(:,state.lastpos)*(1-alpha) + ...
      opt.positions(:,state.nextpos)*alpha;
  ic= 1+mod(round(state.colptr*opt.color_changing_speed), size(opt.color,1));
  state.colptr= state.colptr + 1;
  set(H.patch, 'XData',xy(1,:), 'YData',xy(2,:), 'FaceColor',opt.color(ic,:));
  drawnow;
  pause(1/opt.fps);
  state.step= state.step + 1;
end
