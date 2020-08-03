function [mrk, phrase, symb, final_up_to]= ...
    mrkutil_assignLabelsToOnlineSpellingRSVP(mrk, mrk_cls, varargin)

default_alphabet= ['A':'Z', '_.!<'];

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'idx', 1:length(mrk.pos), ...
                  'alphabet', default_alphabet, ...
                  'target_phrase', [], ...
                  'final_up_to', [], ...
                  'assign_labels', 1);
opt= set_defaults(opt, ...
                  'final_phrase', opt.target_phrase);

idx= opt.idx;
nSelections= numel(mrk_cls.toe);

idx_symb= 1:length(opt.alphabet)*mrk.nRepetitions;
symb= '';
phrase= '';
for ii= 1:numel(mrk_cls.toe),
  selected_symb= mrk_cls.toe(ii)-130;
  
  if opt.assign_labels,
    if ii==1,
      finals= 0;
    else
      finals= opt.final_up_to(ii-1);
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
    elseif strncmp(phrase, opt.target_phrase, length(phrase)),
      % spelled phrase is correct up to here
      % (but parts will get deleted due to accidential backspaces)
      target_symbol= opt.target_phrase(length(phrase)+1);
    else
      target_symbol= '<';
    end
  end
  
  selected_symbol= opt.alphabet(selected_symb);
  symb = [symb selected_symbol];
  if selected_symbol=='<',   % backspace
    if ~isempty(phrase),
      phrase(end)= [];
    end
  else
    phrase= [phrase selected_symbol];
  end
  if opt.assign_labels,
    mrk.y(1,idx(idx_symb))= opt.alphabet(mrk.stimulus(idx(idx_symb)))==target_symbol;
  end
  idx_symb= idx_symb + length(opt.alphabet)*mrk.nRepetitions;
end
mrk.y(2,idx)= 1-mrk.y(1,idx);

if opt.assign_labels & idx_symb(1)<=length(idx),
  mrk= mrk_chooseEvents(mrk, 'not',idx(idx_symb(1):end));
  fprintf('!! Insufficient # classifier outputs: %d events removed.\n',...
          numel(idx)-idx_symb(1)+1);
end

if ~opt.assign_labels,
  final_up_to= zeros(1, length(symb));
  ptr= length(phrase);
  nbacks= 0;
  for k= length(symb):-1:1,
    final_up_to(k)= ptr;
    if symb(k)=='<',
      nbacks= nbacks + 1;
    else
      if nbacks==0,
        ptr= ptr-1;
      else
        nbacks= nbacks - 1;
      end
    end
  end
end
