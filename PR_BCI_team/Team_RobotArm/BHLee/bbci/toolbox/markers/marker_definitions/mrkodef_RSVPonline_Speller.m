function mrk= mrkodef_RSVPonline_Speller(mrko, varargin)

stimDef= {[71:100], [31:70];
          'target','non-target'};
respDef= {'R  1'; 'gTrig'};
miscDef= {252, 253, 200, 201, 105, 106;
          'run_start', 'run_end', 'countdown_start', 'countdown_end', ...
          'burst_start','burst_end'};
c1= num2cell(131:160); 
c2= cprintf('%s',[['A':'Z'] '_.!<'])';
classDef= {c1{:};c2{:}};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'T', [], ...
                  'sbj', '', ...
                  'speller', '', ...
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

mrk.nRepetitions= opt.nRepetitions;

% define classification results:
mrk.classified= mrk_defineClasses(mrko, opt.classDef);

%mrk= mrkutil_removeInvalidBlocks(mrk, 10*opt.nRepetitions);
mrk= mrk_addInfo_P300design(mrk, 30, opt.nRepetitions);
mrk.stimulus= mod(mrk.toe-31,40)+1;

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
  for mm= 1:3,
    mrk.mode(mm,:)= mrk.pos>Tsum(mm) & mrk.pos<=Tsum(mm+1);
    mrk.classified.mode(mm,:)= mrk.classified.pos>Tsum(mm) & ...
        mrk.classified.pos<=Tsum(mm+1);
  end
  mrk= mrk_addIndexedField(mrk, 'mode');
  mrk.classified= mrk_addIndexedField(mrk.classified, 'mode');
  
  mrk= mrkutil_addSpelledPhraseRSVP(mrk);
end

invalid= find(mrk.block_idx==0);
if ~isempty(invalid),
  mrk= mrk_chooseEvents(mrk, 'not',invalid);
  fprintf('%d trials discarded (incomplete blocks)\n', length(invalid));
end
