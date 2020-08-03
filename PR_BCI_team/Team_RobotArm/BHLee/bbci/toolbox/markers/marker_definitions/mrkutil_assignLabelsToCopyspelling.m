function [mrk, phrase, symb, final_up_to]= ...
    mrkutil_assignLabelsToCopyspelling(mrk, mrk_cls, varargin)

default_letter_matrix= ['ABCDE'; 'FGHIJ'; 'KLMNO'; 'PQRST'; 'UVWXY'; 'Z_.,<'];

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'idx', 1:length(mrk.pos), ...
                  'letter_matrix', default_letter_matrix, ...
                  'target_phrase', [], ...
                  'assign_labels', 1, ...
                  'backdoor_bug', 1);

idx= opt.idx;
nSelections= numel(mrk_cls.toe);
if mod(nSelections,2),
  warning('incomplete selection at the end deleted');
  mrk_cls= mrk_chooseEvents(mrk_cls, 'not',nSelections);
end

if ~isfield(mrk, 'correct_selection'),
  mrk.correct_selection= [];
end

idx_level1= 1:6*mrk.nRepetitions;
idx_level2= idx_level1(end) + idx_level1;
symb= '';
phrase= '';
ip= 1;
for ii= 1:2:numel(mrk_cls.toe),
  if ip>length(opt.target_phrase),
    unused= intersect(idx, find(mrk.pos > mrk_cls.pos(ii-1)));
    if ~isempty(unused),
      fprintf('%d events after end of phrase removed.\n', numel(unused));
      mrk= mrk_chooseEvents(mrk, 'not',unused);
      idx= setdiff(idx, unused);
    end
    break;
  end
  selected_level1= mrk_cls.toe(ii)-50;
  selected_level2= mrk_cls.toe(ii+1)-60;
  target_symbol= opt.target_phrase(ip);
  [target_level1, target_level2]= find(target_symbol==opt.letter_matrix);
  if selected_level1~=target_level1,
    target_level2= 6;  % backdoor
  end
  
  if any([selected_level1 selected_level2]<1) | ...
        any([selected_level1 selected_level2]>6),
    error('unexpected classifier output encountered');
  end
  
  if selected_level2==6,  % backdoor
    if ~opt.backdoor_bug,
      ip= ip + 1;
    end
    symb = [symb '^'];
  else
    ip= ip + 1;
    selected_symbol= opt.letter_matrix(selected_level1,selected_level2);
    symb = [symb selected_symbol];
    if selected_symbol=='<',   % backspace
      if ~isempty(phrase),
        phrase(end)= [];
      end
    else
      phrase= [phrase selected_symbol];
    end
  end
  mrk.y(1,idx(idx_level1))= mrk.stimulus(idx(idx_level1))==target_level1;
  mrk.y(1,idx(idx_level2))= mrk.stimulus(idx(idx_level2))==target_level2;
  idx_level1= idx_level1 + 6*2*mrk.nRepetitions;
  idx_level2= idx_level2 + 6*2*mrk.nRepetitions;
  mrk.correct_selection= [mrk.correct_selection, ...
                          target_level1==selected_level1, ...
                          target_level2==selected_level2];
end
mrk.y(2,idx)= 1-mrk.y(1,idx);

if idx_level1(1)<=length(idx),
  mrk= mrk_chooseEvents(mrk, 'not',idx(idx_level1(1):end));
  fprintf('!! Insufficient # classifier outputs: %d events removed.\n',...
          numel(idx)-idx_level1(1)+1);
end
