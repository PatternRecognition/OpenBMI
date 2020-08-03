function hexlm_query(lm, written, varargin)
%HEXLM_QUERY - Visualize probabilities of Hexawrite language model
%
%Synopsis:
% hexlm_query(LM, TEXT, <OPT>)
%
%Arguments:
% LM: Language model, see lm_extractLanguageModel
% TEXT: Text for which probabilities are visualized
% OPT: Struct or property/value list of opiotnal arguements
%  .lm_headfactor: Vector of weights to weight word beginning ('head')
%     probabilities vs within word probabilities. The first value is
%     used as weight for the first letter of a word and so on. The last
%     value of the vector is used as weight for all subsequent letters.
%  .lm_npred: Number of letters for backward probability.
%  .lm_probdelete: Probability of delete symbol, default 0.1.
%
%See:
% lm_extractLanguageModel, lm_loadLanguageModel

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'labelset',...
                   ['ABCDE'; 'FGHIJ'; 'KLMNO'; 'PQRST'; 'UVWXY'; 'Z_<.?'], ...
                  'arrow_reset', 'bestletter', ...
                  'lm_headfactor', [0.85 0.85 0.75 0.5 0.25], ...
                  'lm_letterfactor', 0.01, ...
                  'lm_npred', 2, ...
                  'lm_probdelete', 0.1, ...
                  'gap', 0.05, ...
                  'subplots_vspace', [0.05 0.05 0.02], ...
                  'subplots_hspace', [0.2  0 0.02]);

if ~ismember('<', lm.charset),
  lm.charset= [lm.charset, '<'];
end

written= upper(written);
N= length(written);
clf;
XLim= [0.5 30.5];
for cc= 1:N,
  ax= subplotxl(N, 1, cc, opt.subplots_vspace, opt.subplots_hspace);
  hold on;
  prob= lm_getProbability(lm, written(1:cc-1), opt);
  YLim= [0 min(1, max(prob)*1.05)];
  for hi= 1:6,
    lab= opt.labelset(hi,:);
    for li= 1:5,
      ii= find(lm.charset==lab(li));
      hexprob(li,hi)= prob(ii);
    end
  end
  for hh= 1:6,
    idx= (hh-1)*5+[1:5];
    plot(idx, hexprob(idx), 'k');
    [mm,mi]= max(hexprob(idx));
    plot(idx(mi), hexprob(idx(mi)), '.', 'Color',[0 0.7 0], 'MarkerSize',20);
  end
  lab= opt.labelset';
  ii= find(written(cc)==lab);
  plot(ii, hexprob(ii), 'ro', 'MarkerSize',12, 'LineWidth',2);
  lab= cellstr(lab(:));
  set(ax, 'XTick',1:30, 'XTickLabel',lab);
  set(ax, 'XLim',XLim, 'TickLength',[0 0], 'YGrid','on', 'Box','on');
  line(repmat([0:6]*5+0.5, 2, 1), repmat(YLim', 1, 7), 'Color','k');
  switch(opt.arrow_reset),
   case 'bestletter',
    [mm,mi]= max(hexprob(:));
    settohex= ceil(mi/5);
   case 'besthex',
    [mm,settohex]= max(sum(prob,1));
   otherwise,
    error('policy for arrow_reset not known');
  end
  xx= (settohex-1)*5+[0.5 5.5];
  hp= patch(xx([1 2 2 1]), [0 0 YLim(2)*0.999*[1 1]], [1 0.8 1]);
  moveObjectBack(hp);
  set(ax, 'YLim',YLim);
  ht= text(XLim(1)-opt.gap*diff(XLim), mean(YLim), ...
           written(1:cc-1));
  set(ht, 'HorizontalAli','right', 'VerticalAli','middle', 'FontSize',18);
end
