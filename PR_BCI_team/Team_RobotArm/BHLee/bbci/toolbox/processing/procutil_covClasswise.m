function R= procutil_covClasswise(dat, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'covPolicy', 'average', ...
                  'normalize', 0, ...
                  'weight',ones(1,size(dat.y,2)),...
                  'weight_exp',1);

nClasses= size(dat.y, 1);
nChans= size(dat.x, 2);

%% calculate classwise covariance matrices
R= zeros(nChans, nChans, nClasses);
switch(lower(opt.covPolicy)),
 case 'average',
  for c= 1:nClasses,
    C= zeros(nChans, nChans);
    idx= find(dat.y(c,:));
    for m= idx,
      C= C + (opt.weight(m)^opt.weight_exp)*cov(dat.x(:,:,m));
    end
    R(:,:,c)= C/length(idx);
  end
 case 'normal',
  T= size(dat.x, 1);
  for c= 1:nClasses,
    idx= find(dat.y(c,:));
    x= permute(dat.x(:,:,idx), [1 3 2]);
    x= reshape(x, T*length(idx), nChans);
    R(:,:,c) = cov(x);
  end
 otherwise,
  error('covPolicy not known');
end

t= zeros(nClasses, 1);
if opt.normalize,
  for cc= 1:nClasses,
    t(cc)= trace(R(:,:,cc));
  end
  if opt.normalize==1,
    for cc= 1:nClasses,
      R(:,:,cc)= R(:,:,cc)/t(cc);
    end
  elseif opt.normalize==2,
    for cc= 1:nClasses,
      R(:,:,1)= R(:,:,1)/mean(t);
    end
  else
    error('unknown option for ''normalize''.');
  end
end
