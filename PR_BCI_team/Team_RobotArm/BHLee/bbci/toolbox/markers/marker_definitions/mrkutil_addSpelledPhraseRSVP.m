function mrk= mrkutil_addSpelledPhraseRSVP(mrk, alphabet)

if nargin<2,
  alphabet= ['A':'Z', '_.!<'];
end

mrk.spelledPhrase= {};
mrk.spelledSymbols= {};
mm= strmatch('copyspelling', mrk.modeName);
if isempty(mm),
  warning('no copypelling found');
else
  mrk_cls= mrk_chooseEvents(mrk.classified, find(mrk.classified.mode(mm,:)));
  mrk.spelledSymbols{mm}= alphabet(mrk_cls.toe-130);
  mrk.spelledPhrase{mm}= mrk.spelledSymbols{mm};
end

mm= strmatch('freespelling', mrk.modeName);
if isempty(mm),
  warning('no freespelling found');
  return;
end

idx= find(mrk.mode(mm,:));
mrk_cls= mrk_chooseEvents(mrk.classified, find(mrk.classified.mode(mm,:)));
  
[mrk_dmy, final_phrase, symb, final_up_to]= ...
    mrkutil_assignLabelsToOnlineSpellingRSVP(mrk, mrk_cls, ...
                                             'alphabet', alphabet, ...
                                             'assign_labels', 0);
mrk.spelledPhrase{mm}= final_phrase;
mrk.spelledSymbols{mm}= symb;
if ~isfield(mrk, 'desiredPhrase'),
    mrk.desiredPhrase{mm}= final_phrase;
end
[mrk, phrase, symb]= mrkutil_assignLabelsToOnlineSpellingRSVP(mrk, ...
                                                  mrk_cls, ...
                                                  'alphabet', alphabet, ...
                                                  'target_phrase', mrk.desiredPhrase{mm}, ...
                                                  'final_phrase', final_phrase, ...
                                                  'idx', idx, ...
                                                  'final_up_to', final_up_to);
