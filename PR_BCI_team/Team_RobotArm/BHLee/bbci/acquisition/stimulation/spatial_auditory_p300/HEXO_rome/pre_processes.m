function output = pre_processes(currentState, Lut, Dict, history, varargin),
    %PRE_PROCESSES Summary of this function goes here
    %   Detailed explanation goes here

    opt= propertylist2struct(varargin{:});
    opt= set_defaults(opt, ...
        'sayLabels', true, ...
        'sayResult', true, ...
        'visualize_hexo', 0, ...
        'visualize_text', 1, ...
        'auto_correct', 1, ...
        'auto_target', 1, ...
        'vis_tar_times', 5, ...
        'vis_tar_rate', .2, ...
        'mask_errors', 0, ...
        'fake_feedback', 0, ...
        'no_feedback', 0, ...
        'startup_pause', 0);

    if opt.auto_correct && ~sum([opt.mask_errors, opt.fake_feedback, opt.no_feedback]),
        warning('with auto_correct a simulated feedback should be set. mask_errors is set to true.');
        opt.mask_errors = 1;
    end
    if (opt.no_feedback || opt.fake_feedback || opt.mask_errors) && ~opt.auto_correct,
        error('if fake_feedback, mask_errors or no_feedback should be presented, auto_correct must be true!');
    end
      
    % initialize stuff only the first time
    if ~isfield(opt.procVar, 'handle_background') && ((isfield(opt, 'visualize_hexo') && opt.visualize_hexo) || (isfield(opt, 'visualize_text') && opt.visualize_text)),
        handlefigures('use','user_feedback');
        opt.procVar.handle_background = stimutil_initFigure(opt);
        [opt.procVar.handle_cross, opt.procVar.handle_loc, opt.procVar.handle_label]= stimutil_fixationCrossandLocations(opt, 'angle_offset', 30, 'loc_distance', 0.55, 'loc_radius', .25, 'cross_vpos', -.15, 'label_holders', 1);
        [opt.procVar.copyText opt.procVar.spelledText] = stimutil_addCopyText(opt);
    %     opt.procVar.handles_all = [opt.procVar.handle_loc opt.procVar.handle_cross(1) opt.procVar.handle_cross(2)];    
        set(opt.procVar.handle_cross, 'Visible', 'on');
        if opt.visualize_hexo,
          set(opt.procVar.handle_loc, 'Visible', 'on');
        end
        if ~isfield(opt.procVar, 'graph_rep'),
            opt.procVar.graph_rep = build_graph(Lut);
            disp('Graph representation of spelling tree build.');
        end
        if opt.auto_correct,
            opt.procVar.error_list = zeros(1,length(opt.spellString));
            opt.procVar.trial_error = [];
            opt.procVar.correct_scores = [];
        end
        first_run = 1;
    else
        first_run = 0;
        pause(opt.trial_pause);
    end

    % auto correct a false move if required
    if opt.auto_correct && isfield(opt.procVar, 'trial_target_state') && ~isempty(opt.procVar.trial_target_state)
        currentState = opt.procVar.trial_target_state;
        opt.procVar.currentState = currentState;
    end    
    
    % find next correct move
    if ~opt.auto_correct,
        if isempty(history.written),
            tarChar = opt.spellString(1);
        else
            for i = 1:min(length(history.written), length(opt.spellString)),
                if history.written(i) ~= opt.spellString(i),
                    tarChar = 'delete';
                    break;
                else
                    tarChar = opt.spellString(i+1)
                end
            end
        end
    else
        tarChar = opt.spellString(length(history.written)+1);
    end
    opt.procVar.trial_target = findCorrectStep(opt.procVar.graph_rep, currentState, tarChar);
    opt.procVar.trial_target = opt.procVar.trial_target(1);
    opt.procVar.trial_target_state = Lut(currentState).direction(opt.procVar.trial_target).nState;
    
    % update the written text
    if opt.visualize_text && ~isempty(opt.spellString),
      stimutil_updateCopyText(history.written, opt);drawnow;
      pause(1);
    end
    set(opt.procVar.handle_cross, 'Visible', 'on'); drawnow;

    % show correct labels in the hexograms
    if opt.visualize_hexo,
      update_hexo_labels(currentState, Lut, opt.procVar.handle_label);drawnow;
    end
    
    if first_run,
        pause(opt.startup_pause);
    end

    % indicate the current target, if required
    if opt.auto_target,
        pause(1);
        stimutil_playMultiSound(squeeze(opt.cueStream(opt.procVar.trial_target,:,:))' * opt.calibrated(opt.procVar.trial_target,opt.procVar.trial_target), opt, 'repeat', opt.repeatTarget, 'interval', .3, 'placement', opt.procVar.trial_target);
        origCol = get(opt.procVar.handle_loc(opt.procVar.trial_target), 'FaceColor');
        highlCol = [1 0 0];
        for repI = 1:opt.vis_tar_times,
            set(opt.procVar.handle_loc(opt.procVar.trial_target), 'FaceColor', highlCol);
            pause(opt.vis_tar_rate);
            set(opt.procVar.handle_loc(opt.procVar.trial_target), 'FaceColor', origCol);
            pause(opt.vis_tar_rate);
        end
    end

    % speak out the labels
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
            spchOut{2,i} = opt.cueStream(i,:) * opt.calibrated(i,i);
        end
        spchOut = reshape(spchOut, 1, []);

        stimutil_playMultiSound(spchOut, opt, 'placement', reshape([1:6;1:6],1,[]), 'interval', 0.1);
        pause(0.5);
%     else
%         for i = 1:length(opt.speakerSelected),
%             spchOut{1,i} = opt.cueStream(i,:) * opt.calibrated(i,i);
%         end    
%         stimutil_playMultiSound(spchOut, opt, 'placement', [1:6], 'interval', 0.5);
%         pause(1);
    end

    output = opt.procVar;
end

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
end
 
