function output = post_processes(currentState, Lut, Dict, history, varargin),
%PRE_PROCESSES Summary of this function goes here
%   Detailed explanation goes here

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'sayLabels', true, ...
    'sayResult', true, ...
    'visualize_hexo', 0, ...
    'visualize_text', 1);

if ~isfield(opt.procVar, 'handle_background') && ((isfield(opt, 'visualize_hexo') && opt.visualize_hexo) || (isfield(opt, 'visualize_text') && opt.visualize_text)),
    opt.procVar.handle_background = stimutil_initFigure(opt);
    [opt.procVar.handle_cross, opt.procVar.handle_loc, opt.procVar.handle_label]= stimutil_fixationCrossandLocations(opt, 'angle_offset', 30, 'loc_distance', 0.55, 'loc_radius', .25, 'cross_vpos', -.15, 'label_holders', 1);
    [opt.procVar.copyText opt.procVar.spelledText] = stimutil_addCopyText(opt);
%     opt.procVar.handles_all = [opt.procVar.handle_loc opt.procVar.handle_cross(1) opt.procVar.handle_cross(2)];    
    set(opt.procVar.handle_cross, 'Visible', 'on');
    if opt.visualize_hexo,
      set(opt.procVar.handle_loc, 'Visible', 'on');
    end
end

if opt.visualize_text && ~isempty(opt.spellString),
  stimutil_updateCopyText(history.written, opt);drawnow;
  pause(1);
end
set(opt.procVar.handle_cross, 'Visible', 'on'); drawnow;

if opt.visualize_hexo,
  update_hexo_labels(currentState, Lut, opt.procVar.handle_label);drawnow;
end

if opt.sayLabels,
    labels = cell(1,6);
    [labels{:}] = deal(Lut(currentState).direction.label);

    if isfield(opt.procVar, 'spchHandle'),
        [spchOut] = stimutil_speechSynthesis(labels, opt);
    else
        [spchOut, opt.procVar.spchHandle] = stimutil_speechSynthesis(labels, opt);
    end
    
    for i = 1:length(spchOut),
      spchOut{i} = spchOut{i} * .3;
    end
    
    for i = 1:length(opt.speakerSelected),
%       if isfield(opt, 'targetDir') && opt.targetDir == i,
%         spchOut{2,i} = [opt.cueStream(i,:) * opt.calibrated(i,i) opt.cueStream(i,:) * opt.calibrated(i,i) opt.cueStream(i,:) * opt.calibrated(i,i)];
%       else
        spchOut{2,i} = opt.cueStream(i,:) * opt.calibrated(i,i);
        %pause(.5);
%       end
    end
    spchOut = reshape(spchOut, 1, []);
    
    stimutil_playMultiSound(spchOut, opt, 'placement', reshape([1:6;1:6],1,[]), 'interval', 0.1);
    pause(0.5);
else
    for i = 1:length(opt.speakerSelected),
        spchOut{1,i} = opt.cueStream(i,:) * opt.calibrated(i,i);
        %pause(.5);
%       end
    end    
    stimutil_playMultiSound(spchOut, opt, 'placement', [1:6], 'interval', 0.5);
    pause(1);
end

output = opt.procVar;

function update_hexo_labels(currentState, Lut, handle_label)
  set(handle_label, 'string', '');
  set(handle_label, 'color', [0 0 0]);
  set(handle_label, 'FontSize', 30);
  actionColor = [0 .5 0];

  for ii = 1:size(handle_label,1),
    if length(Lut(currentState).direction(ii).printLabel) > 1,
      for jj = 1:length(Lut(currentState).direction(ii).printLabel),
        set(handle_label(ii, jj+1), 'string', Lut(currentState).direction(ii).printLabel{jj});
        if strmatch(Lut(currentState).direction(ii).type, 'action', 'exact'),
          set(handle_label(ii, jj+1), 'color', actionColor);
        end
        %% do a little hack here
        if strmatch(Lut(currentState).direction(ii).printLabel{jj}, 'Akt', 'exact'),
          set(handle_label(ii, jj+1), 'color', actionColor);
        end
        %% end of hack
        nextState = Lut(currentState).direction(ii).nState;
        if nextState > 1,
          if strmatch(Lut(nextState).direction(jj).type, 'action', 'exact'),
            set(handle_label(ii, jj+1), 'color', actionColor);
          end
        end        
      end
    else
      set(handle_label(ii, 1), 'string', Lut(currentState).direction(ii).printLabel{:});
      if length(Lut(currentState).direction(ii).printLabel{:}) <= 2,
        set(handle_label(ii,1), 'FontSize', 50);
      end
      if strmatch(Lut(currentState).direction(ii).type, 'action', 'exact'),
        set(handle_label(ii, 1), 'color', actionColor);
      end
    end
  end

