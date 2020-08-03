function mrk= mrkodef_MatrixSpeller(mrko, varargin)

stimDef= {[31:36, 41:45], [11:16, 21:25];
          'target','non-target'};
respDef= {'R  1'; 'gTrig'};
miscDef= {240, 241, 246, 247, 250, 251, 254, 255;
          'countdown_start', 'countdown_end', ...
          'feedback_start', 'feedback_end', ...
          'trial_start', 'trial_end', ...
          'run_start', 'run_end'};
%c1= num2cell(131:160); 
%c2= cprintf('%s',[['A':'Z'] '_.!<'])';
%classDef= {c1{:};c2{:}};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'T', [], ...
                  'sbj', '', ...
                  'speller', '', ...
                  'stimDef', stimDef, ...
                  'respDef', respDef, ...
                  'miscDef', miscDef, ...
                  'desiredPhrase', {}, ...
                  'nRows', 6, ...
                  'nRepetitions', 5);
%                  'classDef', classDef, ... % classified element

% define oddball:
mrk= mrkodef_general_oddball(mrko, 'stimDef',opt.stimDef, ...     
                             'respDef',opt.respDef, ...                      
                             'miscDef',opt.miscDef, ...
                             'matchstimwithresp',0);

mrk.nRepetitions= opt.nRepetitions;

% define classification results:
%mrk.classified= mrk_defineClasses(mrko, opt.classDef);

%mrk= mrkutil_removeInvalidBlocks(mrk, 10*opt.nRepetitions);
mrk= mrk_addInfo_P300design(mrk, 11, opt.nRepetitions);
mrk.stimulus= mod(mrk.toe-10,20)+1;
mrk= mrk_addIndexedField(mrk, {'stimulus'});

if isempty(mrk.resp.toe),
  mrk= rmfield(mrk, 'resp');
else
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
  mrk= mrk_addIndexedField(mrk, {'trig_latency', 'pos_trig'});
end

if exist('mrkutil_addSubjectSpecificPhrases','file'),
  mrk= mrkutil_addSubjectSpecificPhrases(mrk, opt.sbj, opt.speller);
else
  mrk.desiredPhrase= opt.desiredPhrase;
end

if isempty(opt.T),
  warning('you didn''t specify T. Thus, mode and spelledPhrase are missing.'),
else
  mrk.T= opt.T;
  Tsum= [0 cumsum(mrk.T)];
  nModes= length(opt.T);
  modeName= {'calibration','copyspelling','freespelling'};
  mrk.modeName= modeName(1:nModes);
%  mrk.classified.modeName= modeName(1:nModes);
  mrk.mode= zeros(nModes, length(mrk.pos));
%  mrk.classified.mode= zeros(nModes, length(mrk.classified.pos));
  for mm= 1:nModes,
    mrk.mode(mm,:)= mrk.pos>Tsum(mm) & mrk.pos<=Tsum(mm+1);
%    mrk.classified.mode(mm,:)= mrk.classified.pos>Tsum(mm) & ...
%        mrk.classified.pos<=Tsum(mm+1);
  end
  mrk= mrk_addIndexedField(mrk, 'mode');
%  mrk.classified= mrk_addIndexedField(mrk.classified, 'mode');
  
%  mrk= mrkutil_addSpelledPhrase(mrk, opt.letter_matrix);
end
