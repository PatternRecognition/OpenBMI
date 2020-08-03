function mrk= mrkodef_ERP_Speller(mrko, varargin)

%     END_LEVEL2 = 245               # end of second hex level
%     COUNTDOWN_START = 240
%     STIMULUS = [ [11, 12, 13, 14, 15, 16] , [21, 22, 23, 24, 25, 26] ]
%     RESPONSE = [ [51, 52, 53, 54, 55, 56] , [61, 62, 63, 64, 65, 66] ]
%     TARGET_ADD = 20
%     ERROR_ADD = 100
%     INVALID_FIXATION = 99

default_letter_matrix= ['ABCDE'; 'FGHIJ'; 'KLMNO'; 'PQRST'; 'UVWXY'; 'Z_.,<'];

stimDef= {[31:46], [11:26];
          'target','nontarget'};
respDef= {'R  1'; 'gTrig'};
miscDef= {252, 253, 240, 244, 245, 99;
          'run_start', 'run_end', 'countdown_start', ...
          'end_level1', 'end_level2', 'invalid'};
classDef = {[51:56 61:66 151:156 161:166];'selected class'};
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'T', [], ...
                  'sbj', '', ...
                  'speller', '', ...
                  'letter_matrix', default_letter_matrix, ...
                  'stimDef', stimDef, ...
                  'respDef', respDef, ...
                  'miscDef', miscDef, ...
                  'classDef', classDef, ... % classified element
                  'ErrPDef', [], ...
                  'nRepetitions', 10);

% define oddball:
mrk= mrkodef_general_oddball(mrko, 'stimDef',opt.stimDef, ...     
                             'respDef',opt.respDef, ...                      
                             'miscDef',opt.miscDef, ...
                             'matchstimwithresp',0);

mrk.letter_matrix= opt.letter_matrix;
mrk.nRepetitions= opt.nRepetitions;

% define classification results:
mrk.classified= mrk_defineClasses(mrko, opt.classDef);
mrk.classified.error = mrk.classified.toe>100;
mrk.classified= mrk_addIndexedField(mrk.classified, 'error');
mrk.classified.toe = mod(mrk.classified.toe, 100);
level= 1 + (mrk.classified.toe>60);

% define ErrPs:
if ~isempty(opt.ErrPDef)
  mrk_ErrP = mrk_defineClasses(mrko, opt.ErrPDef);
  if ~isempty(mrk_ErrP.toe)
    mrk_ErrP = mrk_matchStimWithResp(mrk.classified, mrk_ErrP, ...
                                        'missingresponse_policy', 'accept', ...
                                        'multiresponse_policy', 'first', ...
                                        'max_latency', 2000);
    mrk.classified.ErrP = ~mrk_ErrP.missingresponse;
  else
    mrk.classified.ErrP = zeros(size(mrk.classified.toe));
  end
  mrk.classified= mrk_addIndexedField(mrk.classified, 'ErrP');
  
  % trials with missing level change (e.g. due to eyetracker reset):
  ireject= find(diff(level)==0 & ~mrk.classified.ErrP(1:end-1));
else
  ireject= find(diff(level)==0);
end

if ~isempty(ireject)
  mrk.classified= mrk_chooseEvents(mrk.classified, 'not',ireject);
end

mrk= mrkutil_removeInvalidBlocks(mrk, 6*opt.nRepetitions);
mrk= mrk_addInfo_P300design(mrk, 6, opt.nRepetitions);
mrk.stimulus= mod(mrk.toe-1,10)+1;

if ~isempty(mrk.resp.toe)
  mkk= mrk_matchStimWithResp(mrk, mrk.resp, 'missingresponse_policy','accept',...
                             'multiresponse_policy','first');
  if any(mkk.missingresponse),
    med= nanmedian(mkk.latency);
    imissing= find(mkk.missingresponse);
    mkk.latency(mkk.missingresponse)= med;
    fprintf('%d missing trigger latencies replaced by median.\n', ...
            numel(imissing));
    itoohigh= find(mkk.latency > 1.9*med);
    if ~isempty(itoohigh),
      mkk.latency(itoohigh)= med;
      fprintf('%d latencies that were suspiciously high replaced by median.\n', ...
              numel(itoohigh));
    end
  end
  mrk.trig_latency= mkk.latency;
  mrk.pos_trig= mrk.pos + round(mrk.trig_latency/1000*mrk.fs);
  fprintf('Median latency of trigger: %.1f (range: %g to %g ms; mean %.1f +/- %.1f ms).\n', ...
          median(mkk.latency), min(mkk.latency), max(mkk.latency), ...
          mean(mkk.latency), std(mkk.latency));
else
  mrk.trig_latency= zeros(size(mrk.toe));
  mrk.pos_trig= zeros(size(mrk.toe));
end

mrk= mrk_addIndexedField(mrk, {'stimulus', 'trig_latency', 'pos_trig'});

if exist('mrkutil_addSubjectSpecificPhrases','file'),
  mrk= mrkutil_addSubjectSpecificPhrases(mrk, opt.sbj, opt.speller);
else
  warning('No file with subject specific phrases found.');
end

if isempty(opt.T),
  warning('you didn''t specify T. Thus, mode and spelledPhrase are missing.'),
else
  mrk.T= opt.T;
  Tsum= [0 cumsum(mrk.T)];
  mrk.modeName= {'calibration','copyspelling','freespelling'};
  mrk.classified.modeName= {'calibration','copyspelling','freespelling'};
  mrk.mode= zeros(3, length(mrk.pos));
  mrk.classified.mode= zeros(3, length(mrk.classified.pos));
  for mm= 1:length(mrk.T),
    mrk.mode(mm,:)= mrk.pos>Tsum(mm) & mrk.pos<=Tsum(mm+1);
    mrk.classified.mode(mm,:)= mrk.classified.pos>Tsum(mm) & ...
        mrk.classified.pos<=Tsum(mm+1);
  end
  mrk= mrk_addIndexedField(mrk, 'mode');
  mrk.classified= mrk_addIndexedField(mrk.classified, 'mode');
  
  mrk= mrkutil_addSpelledPhrase(mrk, opt.letter_matrix);
end
