function mrk= mrkutil_removeInvalidBlocks(mrk, expected_length)

warning('off', 'all')
mrk_invalid= mrk_selectClasses(mrk.misc, 'invalid');
mrk_level_bounds = mrk_selectClasses(mrk.misc, {'run_start', ...
                                                'run_end', ...
                                                'countdown_start', ...
                                                'end_level1', ...
                                                'end_level2'});
warning('on', 'all')

%%
red_list= [];
ptr= 0;
level= 1 + (mod(mrk.toe-11, 20)>=10);
while ptr<length(mrk.pos),
  % find level ival:
  level_start= ptr+1;
  current_level= level(level_start);
  level_length= find([level(level_start:end) NaN]~=current_level, 1)-1;
  ival= mrk.pos([level_start level_start+level_length-1]);
  
  % check for invalid trial:
  invalid_trial = find(mrk_invalid.pos>=ival(1) & mrk_invalid.pos<=ival(2), 1);
  
  % check for level bounds during the ival:
  bound_pos = mrk_level_bounds.pos(find(mrk_level_bounds.pos>ival(1) & mrk_level_bounds.pos<ival(2),1));
  if ~isempty(bound_pos)
    level_length = length(find(mrk.pos>=ival(1) & mrk.pos<bound_pos));
  end
  
  % check for level length:
  if level_length > expected_length,
    % This should never happen anymore. Invalid trials should be splitted into one
    % short level (which then is detected to be too short) and one normal level.
    fprintf('level (#%d) starting at %.0f ms (marker #%d) is too long (length %d). truncating...', ...
            current_level, mrk.pos(level_start)/mrk.fs*1000, level_start, level_length);
    if isempty(invalid_trial),
      fprintf(' NO invalid marker.');
    end
    fprintf('\n');
    red_list= [red_list, level_start+[0:level_length-expected_length-1]];
  elseif level_length < expected_length || ~isempty(invalid_trial),
    % If a level #2 is too short, we would need to delete preceding #1, too ???
    fprintf('level (#%d) starting at %.0f ms (marker #%d) is too short (length %d). deleting...', ...
            current_level, mrk.pos(level_start)/mrk.fs*1000, level_start, level_length);
    if isempty(invalid_trial),
      fprintf(' NO invalid marker.');
    end
    fprintf('\n');
    red_list= [red_list, level_start:level_start+level_length-1];
  end
  ptr= ptr + level_length;
end
%%
mrk= mrk_chooseEvents(mrk, 'not',red_list);
