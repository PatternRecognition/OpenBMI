function [newState, selection, output] = post_processes(clOut, currentState, Lut, history, varargin);
%POST_PROCESSES Summary of this function goes here
%   Detailed explanation goes here

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'sayLabels', true, ...
    'sayResult', true, ...
    'space_char', '_', ...
    'errorPTrig', 200, ...
    'errorPRec', true, ...
    'visualize_result', 1, ...
    'vis_result_times', 5, ...
    'vis_result_rate', .2);

if ~isfield(opt.procVar, 'spchHandle'),
    opt.procVar.spchHandle = [];
end

if clOut.class_label == -10,
    newState = -10;
    output = opt.procVar;
    selection = -10;
    return;
else
    newState = Lut(currentState).direction(clOut.class_label).nState;
end

if opt.auto_correct && isfield(opt.procVar, 'trial_target'),
    save_dir = clOut.class_label;
    clOut.class_label = opt.procVar.trial_target;
end

switch Lut(currentState).direction(clOut.class_label).type,
    case 'navi'
        selection = [];
    case 'select'
        selection = Lut(currentState).direction(clOut.class_label).alt;

        fprintf('Letter written: %s\n', Lut(currentState).direction(clOut.class_label).label);
        
        switch Lut(currentState).direction(clOut.class_label).alt,
            case {'?', '.'}
                [sentence, flag] = get_Parts([history.written Lut(currentState).direction(clOut.class_label).alt], '[.?]', 'sentence');
                if flag & ~opt.auto_correct,
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis({Lut(1).speech.sentenceWrite, sentence}, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1], 'interval', .5);
                end

            case {' ', ','},
                if ~isempty(history.written) && ~ismember(history.written(end), {' ', '.', ',', '?'})  & ~opt.auto_correct,
                    [word, flag] = get_Parts([history.written Lut(currentState).direction(clOut.class_label).alt], '[\s,.?]', 'last');
                    if flag,
                        [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis({Lut(1).speech.wordWrite, word}, opt);
                        stimutil_playMultiSound(saySound, opt, 'placement', [1], 'interval', .5);
                    end  
                end
        end   
            
    case 'action'
        fprintf('Perform action: %s\n', Lut(currentState).direction(clOut.class_label).label);
        switch Lut(currentState).direction(clOut.class_label).alt,
            case 'delete'
                selection = -1;
                
            case 'W_spell'
                [word, flag] = get_Parts(history.written, '[\s.,:?]', 'last');
                if flag,
                    letters = {};
                    for i = 1:length(word{1}),
                        [letters(i), opt.procVar.spchHandle] = stimutil_speechSynthesis(word{1}(i), opt);
                    end
                    stimutil_playMultiSound(letters, opt, 'placement', [1], 'interval', 0.3);
                else 
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis(Lut(1).speech.noText, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1]);
                end
                selection = [];
                
            case 'W_read'
                [word, flag] = get_Parts(history.written, '[\s.,:?]', 'last');
                if flag,
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis(word, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1]);
                else 
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis(Lut(1).speech.noText, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1]);
                end          
                selection = [];
                
            case 'S_read'
                [sentence, flag] = get_Parts(history.written, '[.?]', 'sentence');
                if flag,
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis(sentence, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1]);
                else 
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis(Lut(1).speech.noText, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1]);
                end   
                selection = [];
                
            case 'T_read'
                [dmy, flag] = get_Parts(history.written, '[.?]', 'sentence');
                if flag,
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis(history.written, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1]);
                else 
                    [saySound, opt.procVar.spchHandle] = stimutil_speechSynthesis(Lut(1).speech.noText, opt);
                    stimutil_playMultiSound(saySound, opt, 'placement', [1]);
                end          
                selection = [];

            otherwise
                selection = [];
        end
end

if opt.auto_correct && isfield(opt.procVar, 'trial_target'),
    clOut.class_label = save_dir;
end
% report the result
if opt.errorPRec,
    pause(1);
    ppTrigger(opt.errorPTrig);
    stimutil_playMultiSound({opt.cueStream(clOut.class_label,:) * opt.calibrated(clOut.class_label,clOut.class_label)}, opt, 'placement', clOut.class_label);
else
  pause(1.5);    
end

if opt.visualize_result,
    origCol = get(opt.procVar.handle_loc(clOut.class_label), 'FaceColor');
    highlCol = [.8 0.8 0];
    for repI = 1:opt.vis_result_times,
        set(opt.procVar.handle_loc(clOut.class_label), 'FaceColor', highlCol);
        pause(opt.vis_result_rate);
        set(opt.procVar.handle_loc(clOut.class_label), 'FaceColor', origCol);
        pause(opt.vis_result_rate);
    end
end

if opt.sayResult && ~opt.auto_correct,
    textToSay = [Lut(currentState).direction(clOut.class_label).label ' ' Lut(1).speech.selected];
    
    [spchOut, opt.procVar.spchHandle] = stimutil_speechSynthesis(textToSay, opt);
    for i = 1:length(spchOut),
      spchOut{i} = spchOut{i} * .3;
    end    
  
    stimutil_playMultiSound(spchOut, opt, 'placement', clOut.class_label, 'interval', 0.5);
    pause(1); 
end

% update error history
if opt.auto_correct,
    opt.procVar.trial_error = [opt.procVar.trial_error, (clOut.class_label ~= opt.procVar.trial_target)];
    opt.procVar.correct_scores = [opt.procVar.correct_scores, (clOut.class_label == opt.procVar.trial_target)];
    if opt.procVar.trial_target_state == 1,
        opt.procVar.error_list(length(history.written)+1) = bin2dec(num2str(opt.procVar.trial_error));
        opt.procVar.trial_error = [];
    end
end


% display the result
set(opt.procVar.spelledText, 'string', '');
if ~isempty(selection) && selection ~= -1,
  writtenString = regexprep(upper([history.written selection]), ' ', opt.space_char);
else
  writtenString = regexprep(upper(history.written), ' ', opt.space_char);
end
stimutil_updateCopyText(writtenString, opt); drawnow;

output = opt.procVar;
end

function [words, flag] = get_Parts(strSeq, mtchExp, whichWords),
    words = {};
    flag = 0;

    if isempty(strSeq) || length(regexp(strSeq, mtchExp, 'match'))==length(strSeq),
        return;
    end
    
    switch whichWords
        case 'sentence'
            idx = regexp(strSeq, mtchExp);
            textParts = {};
            start = 1;
            if isempty(idx),
                idx = length(strSeq);
            elseif idx(end) < length(strSeq),
                idx(end+1) = length(strSeq);
            end
            for i=1:length(idx),
%                 if start == idx(i);break;end
                textParts{i} = strSeq(start:idx(i));
                start = idx(i)+1;
                while start <= length(strSeq) && ismember(strSeq(start), mtchExp(2:end-1)),
                    start = start+1;
                end
            end
            textParts(cellfun('isempty', textParts)) = [];
            textParts = strtrim(textParts); 
        otherwise
            textParts = regexp(strSeq, mtchExp, 'split');
            textParts(cellfun('isempty', textParts)) = [];
            if isempty(textParts);textParts{1} = strSeq;end
    end

    switch whichWords
        case 'last'
            words = textParts(end);
            flag = 1;
        case 'first'
            words = textParts(1);
            flag = 1;
        case 'sentence'
            words = textParts(end);
            flag = 1;
        otherwise
            words = textParts;
            flag = 1;
    end
end
