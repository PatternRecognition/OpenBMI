function proj= procutil_biplist2projection(clab, bip_list, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'delete_policy', 'auto', ...
                  'label_policy', 'auto');

if isfield(clab, 'clab'),  %% if first argument is, e.g.,  cnt or epo struct
  clab= clab.clab;
end

for bb= 1:length(bip_list),
  bip= bip_list{bb};
  proj(bb).chan= bip{1};
  proj(bb).filter= zeros(length(clab), 1);
  proj(bb).filter(chanind(clab, bip{1:2}))= [1 -1];
  if length(bip)>2,
    proj(bb).new_clab= bip{3};
  else
    switch(lower(opt.label_policy)),
     case 'auto',
      if strcmpi(bip{2}(end-2:end), 'ref'),
        proj(bb).new_clab= clab{proj(bb).cidx};
      elseif ismember(bip{1}(end), 'pn') & ismember(bip{2}(end),'pn') | ...
             ismember(bip{1}(end), 'lr') & ismember(bip{2}(end),'lr'),
        proj(bb).new_clab= bip{1}(1:end-1);
      end
     case 'deletelastchar',
      proj(bb).new_clab= bip{1}(1:end-1);
     case 'firstlabel',
      proj(bb).new_clab= clab{proj(bb).cidx};
     otherwise,
      error('choice for OPT.label_policy unknown');
    end
  end
  switch(lower(opt.delete_policy)),
   case 'auto',
    if numel(bip{2})>2 && strcmpi(bip{2}(end-2:end), 'ref') | ...
          (ismember(bip{1}(end), 'pn') & ismember(bip{2}(end),'pn')) | ...,
          (ismember(bip{1}(end), 'lr') & ismember(bip{2}(end),'lr')),
      proj(bb).rm_clab= bip{2};
    else
      proj(bb).rm_clab= {};
    end
   case 'second',
    proj(bb).rm_clab= bip{2};
   case 'never',
    proj(bb).rm_clab= {};
   otherwise,
    error('choice for OPT.delete_policy unknown');
  end
end

[proj.clab]= deal(clab);
