function [mrk, phrase, symb, final_up_to]= ...
    mrkutil_assignLabelsToOnlineSpelling(mrk, mrk_cls, varargin)

default_letter_matrix= ['ABCDE'; 'FGHIJ'; 'KLMNO'; 'PQRST'; 'UVWXY'; 'Z_.,<'];

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'idx', 1:length(mrk.pos), ...
                  'letter_matrix', default_letter_matrix, ...
                  'target_phrase', [], ...
                  'final_up_to', [], ...
                  'assign_labels', 1, ...
                  'copyspelling', 0);
opt= set_defaults(opt, ...
                  'final_phrase', opt.target_phrase);

idx= opt.idx;
nSelections= numel(mrk_cls.toe);
if mod(nSelections,2),
  warning('incomplete selection at the end deleted');
  mrk_cls= mrk_chooseEvents(mrk_cls, 'not',nSelections);
end

if ~isfield(mrk, 'correct_selection') && opt.assign_labels,
  mrk.correct_selection= [];
end

idx_level1= 1:6*mrk.nRepetitions;
idx_level2= idx_level1(end) + idx_level1;
symb= '';
phrase= '';
for ii= 1:2:numel(mrk_cls.toe),
  selected_level1= mrk_cls.toe(ii)-50;
  selected_level2= mrk_cls.toe(ii+1)-60;
  
  if opt.assign_labels,
    if ii==1,
      finals= 0;
    else
      finals= opt.final_up_to((ii-1)/2);
    end
    if finals==length(phrase),  % spelled phrase is final up to here
      if finals==length(opt.target_phrase),
        unused= intersect(idx, find(mrk.pos > mrk_cls.pos(ii-1)));
        if ~isempty(unused),
          fprintf('%d events after end of phrase removed.\n', numel(unused));
          mrk= mrk_chooseEvents(mrk, 'not',unused);
          idx= setdiff(idx, unused);
        end
        break;
      end
      target_symbol= opt.target_phrase(finals+1);
      [target_level1, target_level2]= find(target_symbol==opt.letter_matrix);
    elseif strncmp(phrase, opt.target_phrase, length(phrase)),
      % spelled phrase is correct up to here
      % (but parts will get deleteed due to accidential backspaces)
      target_symbol= opt.target_phrase(length(phrase)+1);
      [target_level1, target_level2]= find(target_symbol==opt.letter_matrix);
    else
      [target_level1, target_level2]= find('<'==opt.letter_matrix);  % backspc
    end
    if selected_level1~=target_level1,
      target_level2= 6;  % backdoor
    end
  end
  
  if any([selected_level1 selected_level2]<1) | ...
        any([selected_level1 selected_level2]>6),
    error('unexpected classifier output encountered');
  end
  
  if selected_level2==6,  % backdoor
    symb = [symb '^'];
  else
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
  if opt.assign_labels,
    mrk.y(1,idx(idx_level1))= mrk.stimulus(idx(idx_level1))==target_level1;
    mrk.y(1,idx(idx_level2))= mrk.stimulus(idx(idx_level2))==target_level2;
    mrk.correct_selection= [mrk.correct_selection, ...
                            target_level1==selected_level1, ...
                            target_level2==selected_level2];
  end
  idx_level1= idx_level1 + 6*2*mrk.nRepetitions;
  idx_level2= idx_level2 + 6*2*mrk.nRepetitions;
end
mrk.y(2,idx)= 1-mrk.y(1,idx);

if opt.assign_labels & idx_level1(1)<=length(idx),
  mrk= mrk_chooseEvents(mrk, 'not',idx(idx_level1(1):end));
  fprintf('!! Insufficient # classifier outputs: %d events removed.\n',...
          numel(idx)-idx_level1(1)+1);
end

if ~opt.assign_labels,
  final_up_to= zeros(1, length(symb));
  ptr= length(phrase);
  nbacks= 0;
  for k= length(symb):-1:1,
    final_up_to(k)= ptr;
    if symb(k)=='<',
      nbacks= nbacks + 1;
    elseif symb(k)~='^',
      if nbacks==0,
        ptr= ptr-1;
      else
        nbacks= nbacks - 1;
      end
    end
  end
end
