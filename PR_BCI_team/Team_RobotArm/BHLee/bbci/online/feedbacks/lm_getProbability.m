function prob= lm_getProbability(lm, written, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'lm_headfactor', [0.85 0.85 0.75 0.5 0.25], ...
                  'lm_letterfactor', 0.01, ...
                  'lm_npred', 2);

if isempty(written),
  head= '';
else
  idx= max(find(written=='_')) + 1;
  if isempty(idx),
    idx= 1;
  end
  head= written(idx:end);
end
wl= length(head);
if wl<length(lm.head_table),
  idx= strmatch(head, lm.head_table{wl+1}, 'exact');
else
  idx= [];
end
if isempty(idx),
  hp= 1/(length(lm.charset)-1);
else
  hp= lm.head_prob{wl+1}(:,idx);
end
bb= min(wl, opt.lm_npred) + 1;
idx= [];
while isempty(idx),
  bb= bb - 1;
  idx= strmatch(written(end-bb+1:end), lm.pred_table{bb+1}, 'exact');
end
pp= lm.pred_prob{bb+1}(:,idx);
hf= opt.lm_headfactor(min([wl+1 length(opt.lm_headfactor)]));
prob= hp*hf + pp*(1-hf);
if bb>0,
  prob= lm.pred_prob{1}*opt.lm_letterfactor + prob*(1-opt.lm_letterfactor);
end

if isfield(opt, 'lm_probdelete'),
  prob= [prob*(1-opt.lm_probdelete); opt.lm_probdelete];
end
