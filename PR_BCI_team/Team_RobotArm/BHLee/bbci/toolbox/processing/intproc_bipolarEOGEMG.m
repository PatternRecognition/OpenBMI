function cnt= intproc_bipolarEOGEMG(cnt, varargin)

bip_list= {{'EOGvp','EOGvn'}, ...
           {'EOGhp','EOGhn'}, ...
           {'EMGlp','EMGln'}, ...
           {'EMGrp','EMGrn'}, ...
           {'EMGfp','EMGfn'}};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'delete_channels', 1, ...
                  'bip_list', bip_list);

tobedeleted= [];
for bb= 1:length(opt.bip_list),
  chans= opt.bip_list{bb};
  if ~ismember(chans{1}, cnt.clab),
    warning(sprintf('channel <%s> not found.', chans{1}));
    continue
  end
  cnt= make_bipolar(cnt, chans, opt);
  tobedeleted= [tobedeleted, clabindices(cnt, chans(2), opt)];
end

if opt.delete_channels,
  cnt.x(:,tobedeleted)= [];
  cnt.clab(tobedeleted)= [];
  if isfield(cnt, 'scale'),
    cnt.scale(tobedeleted)= [];
  end
else
  cnt.clab(tobedeleted)= {'NaC'};
end



function cnt= make_bipolar(cnt, chans, opt)

if length(chans)<3,
  chans{3}= chans{1}(1:end-1);
end

cidx= clabindices(cnt, chans(1:2), opt);
cnt.x(:,cidx(1))= int16(double(cnt.x(:,cidx(1:2)))*[1; -1]);
cnt.clab{cidx(1)}= chans{3};
