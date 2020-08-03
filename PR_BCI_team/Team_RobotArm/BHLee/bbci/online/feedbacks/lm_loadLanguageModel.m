function lm= lm_loadLanguageModel(file)

global DATA_DIR

lm= load([DATA_DIR 'language/' file]);
nChars= size(lm.head_prob{1},1);
lm.head_prob{1}= lm.head_prob{1} / sum(lm.head_prob{1},1);
for hl= 2:length(lm.head_prob),
  lm.head_prob{hl}= lm.head_prob{hl} ./ ...
      (ones(nChars,1)*sum(lm.head_prob{hl},1));
end
lm.pred_prob{1}= lm.pred_prob{1} / sum(lm.pred_prob{1},1);
for hl= 2:length(lm.pred_prob),
  lm.pred_prob{hl}= lm.pred_prob{hl} ./ ...
      (ones(nChars,1)*sum(lm.pred_prob{hl},1));
end
