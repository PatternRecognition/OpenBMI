function [fvcspp, varargout] = proc_cspp_auto(dat, varargin)

opt = propertylist2struct(varargin{:});

opt= set_defaults(opt, ...
  'patch_selectPolicy', 'directorscut', ...
  'patch_score', 'medianvar', ...
  'nPatPerPatch', 1, ...
  'csp_selectPolicy', 'equalperclass', ...
  'csp_score', 'medianvar', ...
  'patterns', 3, ...
  'covPolicy', 'average', ...
  'patch','twelve', ...
  'patch_centers', '*', ...
  'patch_clab', [], ...
  'require_complete_neighborhood', 1);

if length(dat) == 1  
  if isfield(dat, 'origClab')
    origClab = dat.origClab;
    nBands = length(dat.clab)/length(origClab);
  else
    origClab = dat.clab;        
    nBands = 1;
  end
else    
  origClab = dat{1}.origClab;
  nBands = 1;
end

opt.patch_centers= origClab(chanind(origClab,opt.patch_centers));

nChans = length(origClab);
nCenters = length(opt.patch_centers);

if isempty(opt.patch_clab)    
  allch = scalpChannels;
  [dummy1 dummy2 neighborClabs] = getClabForLaplacian(allch, 'clab', opt.patch_centers, ...
    'filter_type', opt.patch, 'require_complete_neighborhood', opt.require_complete_neighborhood);
  opt.patch_clab = cell(1,nCenters);
  center_to_rm = [];
  for ic = 1:nCenters
    if ~isempty(neighborClabs{ic})
      opt.patch_clab{ic} = cat(2,opt.patch_centers{ic}, neighborClabs{ic});
    else
      center_to_rm = [center_to_rm ic];
    end
  end            
  opt.patch_centers(center_to_rm) = [];
  nCenters = length(opt.patch_centers);
  opt.patch_clab(center_to_rm) = [];
end

patch_and_pat_idx = [];
W = [];
eig_patch = [];
Wclab = cell(1,1);

area_LH = {'FFC5-3','FC3','CFC5-3','C5-3','CCP5-3','CP5-3','PCP5-3'};
area_C = {'FFC1-2','FC1-2','CFC1-2','C1-2','CCP1-2','CP1-2','PCP1-2','P1-2'};
%area_C = {'FFC1-2','FCz','CFC1-2','Cz','CCP1-2','CPz','PCP1-2','Pz'};
area_RH = {'FFC4-6','FC4','CFC4-6','C4-6','CCP4-6','CP4-6','PCP4-6'};
default_motorarea = {area_LH, area_RH, area_C};

opt.motorarea = default_motorarea;

for ib = 1:nBands
  
  if nBands == 1
    suff = '';
  else
    suff = ['_flt' int2str(ib)];
  end
  
  for ic = 1:nCenters
    
    idxclabs = (ib-1)*nChans+chanind(origClab, opt.patch_clab{ic});
    
    if length(dat) == 1
      fv = proc_selectChannels(dat, idxclabs);
    else
      fv = dat{ic};
    end       
  
    if ~isstr(opt.covPolicy)
      R = opt.covPolicy(idxclabs,idxclabs,:);
    else
      R = opt.covPolicy;
    end
  
    if length(fv.clab) >= 2
      if opt.nPatPerPatch <= floor(length(fv.clab)/2)
        [fv, w_tmp, eig] = proc_csp_auto(fv, 'patterns', opt.nPatPerPatch, ...
          'selectPolicy', opt.csp_selectPolicy, 'score', opt.csp_score, 'covPolicy', R);
      else
        nPatPerPatch = floor(length(fv.clab)/2);
        [fv, w_tmp, eig] = proc_csp_auto(fv, 'patterns', nPatPerPatch, ...
          'selectPolicy', opt.csp_selectPolicy, 'score', opt.csp_score, 'covPolicy', R);
      end
    
      w = zeros(nChans*nBands, size(w_tmp,2));
      w(idxclabs,:) = w_tmp;
      W = cat(2,W,w);
    
      for ip = 1:size(w,2)     
        Wclab(end+1) = opt.patch_centers(ic);                
%         fv.clab{ip} = ['patch' opt.patch_centers{ic} suff '-Pat' int2str(ip)];
        patch_and_pat_idx = cat(1, patch_and_pat_idx, [(ib-1)*nCenters+ic ip]);
      end
    
      if ic == 1 && ib == 1
        fvcspp = fv;
      else
        fvcspp = proc_appendChannels(fvcspp, fv);
      end
      
      eig_patch = cat(1, eig_patch, eig);
    end    
  end
end

if ~isempty(W)
  maxPatterns = size(W,2);
  
  switch opt.patch_score
    case 'eigenvalues',
      score = eig_patch;
    case 'medianvar',
      %       fv = proc_linearDerivation(dat, W);
      fv = proc_variance(fvcspp);
      score = zeros(maxPatterns, 1);
      c1 = find(fv.y(1,:));
      c2 = find(fv.y(2,:));
      for kk = 1:maxPatterns,
        v1 = median(fv.x(1,kk,c1),3);
        v2 = median(fv.x(1,kk,c2),3);
        score(kk) = v2/(v1+v2);
      end
    case 'roc',
      %       fv = proc_linearDerivation(dat, W);
      fv = proc_variance(fvcspp);
      fv = proc_rocAreaValues(fv);
      score = -fv.x;
    case 'fisher',
      %       fv = proc_linearDerivation(dat, W);
      fv = proc_variance(fvcspp);
      fv = proc_logarithm(fv);
      fv = proc_rfisherScore(fv, 'preserve_sign', 1);
      score= -fv.x;
    case 'fsdd'
      %       fv = proc_linearDerivation(dat, W);
      fv = proc_variance(fvcspp);
      fv = proc_logarithm(fv);
      fv = proc_flaten(fv);
      D = fv.x';
      [b,i,j]=unique(fv.y(1,:));
      for k = 1:length(b)
        n(k,1) = sum(j==k);
        m(k,:) = mean(D(j==k,:),1);
        v(k,:) = var(D(j==k,:),1);
      end
      m0 = mean(m,1,n);
      v0 = var(D,[],1);
      s2 = mean(m.^2,1,n) - m0.^2;
      score = (s2 - 2*mean(v,1,n)) ./ v0;
      [t,idx] = sort(-score);
      score = t;
    case 'rank'
      %       fv = proc_linearDerivation(dat, W);
      fv = proc_variance(fvcspp);
      fv = proc_logarithm(fv);
      fv = proc_flaten(fv);
      D = fv.x';
      N = size(D,2);
      score = repmat(NaN,1,N);
      %       cl = cat2bin(fv.y(1,:));
      cl = fv.y';
      [tmp,D] = sort(D,1);
      idx = repmat(NaN,1,N);
      for k = 1:N,
        f = isnan(score);
        X = cl; Y = D(:,f); Z = D(:,~f);
        if (k>1)
          X = X-Z*(Z\X);
          Y = Y-Z*(Z\Y);
        end;
        r = corrcoefnan(X,Y);
        [s,ix] = max(sumsq(r,1));
        f = find(f);
        idx(k) = f(ix);
        score(idx(k)) = s;
      end
    case 'Pearson'
      %       fv = proc_linearDerivation(dat, W);
      fv = proc_variance(fvcspp);
      fv = proc_logarithm(fv);
      fv = proc_flaten(fv);
      D = fv.x';
      N = size(D,2);
      score = repmat(NaN,1,N);
      %       cl = cat2bin(fv.y(1,:));
      cl = fv.y';
      idx = repmat(NaN,1,N);
      for k = 1:N,
        f = isnan(score);
        X = cl; Y = D(:,f); Z = D(:,~f);
        if (k>1)
          X = X-Z*(Z\X);
          Y = Y-Z*(Z\Y);
        end;
        r = corrcoefnan(X,Y);
        [s,ix] = max(sumsq(r,1));
        f = find(f);
        idx(k) = f(ix);
        score(idx(k)) = s;
      end
      
    otherwise,
      error('unknown option for score');
  end
  
  %% force score to be a column vector
  score= score(:);
  
  %% select patterns
  switch opt.patch_selectPolicy
    case 'all'
      fi= 1:maxPatterns;
    case 'auto'
      perc= percentiles(score, [20 80]);
      thresh= perc(2) + diff(perc);
      fi= find(score>thresh);
    case 'automaxvalues'
      score= max(score, 1-score);
      perc= percentiles(score, [20 80]);
      thresh= perc(2) + diff(perc);
      fi= find(score>thresh);
    case 'equalperclass'
      [dd,di]= sort(score);
      fi = [di(1:min(floor(maxPatterns/2),opt.patterns)); di(end:-1:maxPatterns-min(floor(maxPatterns/2),opt.patterns)+1)];
    case 'equalperclass2'
      absscore= 2*(max(score, 1-score)-0.5);
      [dd,di]= sort(absscore);
      fi = [di(1:min(floor(maxPatterns/2),opt.patterns)); di(end:-1:maxPatterns-min(floor(maxPatterns/2),opt.patterns)+1)];
    case 'bestvalues'
      [dd,di]= sort(min(score, 1-score));
      fi= di(1:opt.patterns);
    case 'bestvaluesPerArea'
      nAreas = length(opt.motorarea);
      nPatPerArea = floor(opt.patterns/nAreas);
      [dd,di]= sort(min(score, 1-score));
      clab = Wclab(di);
      fi = [];
      for ii = 1:nAreas
        idx_area = chanind(clab, opt.motorarea{ii});
        fi = cat(2,fi,di(idx_area(1:nPatPerArea)));
      end
    case 'maxvalues2',
      absscore= 2*(max(score, 1-score)-0.5);
      [dd,di]= sort(-absscore);
      fi= di(1:min(maxPatterns,opt.patterns));
    case 'maxvalues'
      [dd,di]= sort(-score);
      fi= di(1:min(maxPatterns,opt.patterns));
    case 'maxvaluesPerArea'
      nAreas = length(opt.motorarea);
      nPatPerArea = floor(opt.patterns/nAreas);
      [dd,di]= sort(-score);
      clab = Wclab(di);
      fi = [];
      for ii = 1:nAreas,
        idx_area = chanind(clab, opt.motorarea{ii});
        fi = cat(2,fi,di(idx_area(1:nPatPerArea)));
      end
    case 'maxvalueswithcut'
      score= score/max(score);
      [dd,di]= sort(-score);
      iMax= 1:opt.patterns;
      iCut= find(-dd>=0.5);
      idx= intersect(iMax, iCut);
      fi= di(idx);
    case 'maxvalueswithcutPerArea'
      nAreas = length(opt.motorarea);
      nPatPerArea = floor(opt.patterns/nAreas);
      score= score/max(score);
      [dd,di]= sort(-score);
      clab = Wclab(di);
      fi = [];
      for ii = 1:nAreas,
        idx_area = chanind(clab, opt.motorarea{ii});
        score_area = dd(idx_area);
        iMax= 1:nPatPerArea;
        iCut= find(-score_area>=0.5);
        idx= intersect(iMax, iCut);
        fi = cat(2,fi,di(idx_area(idx)));
      end
    case 'directorscut',
      if ismember(opt.patch_score, {'eigenvalues','medianvar'}),
        absscore= 2*(max(score, 1-score)-0.5);
        [dd,di]= sort(score);
        Nh= floor(maxPatterns/2);
        iC1= find(ismember(di, 1:Nh));
        iC2= flipud(find(ismember(di, maxPatterns-Nh+1:maxPatterns)));
        iCut= find(absscore(di)>=0.66*max(absscore));
        idx1= [iC1(1); intersect(iC1(2:min(opt.patterns,length(iC1))), iCut)];
        idx2= [iC2(1); intersect(iC2(2:min(opt.patterns,length(iC2))), iCut)];
        fi= di([idx1; flipud(idx2)]);
      else
        score= score/max(score);
        [dd,di]= sort(-score);
        Nh= floor(maxPatterns/2);
        iC1= find(ismember(di, 1:Nh));
        iC2= find(ismember(di, maxPatterns-Nh+1:maxPatterns));
        iCut= find(-dd>=0.5);
        idx1= [iC1(1); intersect(iC1(2:min(opt.patterns,length(iC1))), iCut)];
        idx2= [iC2(1); intersect(iC2(2:min(opt.patterns,length(iC2))), iCut)];
        fi= di([idx1; idx2]);
      end
    case 'matchfilters'
      fi= zeros(1,size(opt.patterns,2));
      for ii= 1:size(opt.patterns,2),
        v1= opt.patterns(:,ii);
        v1= v1/sqrt(v1'*v1);
        sp= -inf*ones(1,maxPatterns);
        for jj= 1:maxPatterns,
          if ismember(jj, fi)
            continue; 
          end
          v2= W(:,jj);
          sp(jj)= abs(v1'*v2/sqrt(v2'*v2));
        end
        [mm,mi]= max(sp);
        fi(ii)= mi;
      end      
    otherwise,
      error('unknown selectPolicy');
  end
  
  Wp= W(:,fi);
  la= score(fi);
  usedPat = zeros(length(fi), 3);
  for ic = 1:length(fi)    
    usedPat(ic,1) = fi(ic);
    usedPat(ic,2:3) = patch_and_pat_idx(fi(ic),:);
    clab{ic} = ['patch' opt.patch_centers{usedPat(ic,2)} suff '-Pat' int2str(usedPat(ic,3))];        
  end
  
  fvcspp = proc_selectChannels(fvcspp, fi);
  fvcspp.clab= clab;
  
  if nBands == 1
    fvcspp.origClab = origClab;
  else
    fvcspp.origClab = dat.origClab;
  end
  
  %% arrange optional output arguments
  if nargout>1,
    varargout{1}= Wp;
    if nargout>2,
      varargout{2}= la;
      if nargout>3,
        A= pinv(W);
        varargout{3}= A(fi,:);
        if nargout>4,
          varargout{4}= usedPat;
          if nargout>5
            varargout{5} = W;
          end
        end
      end
    end
  end
else
  varargout{:} = [];
  fvcspp = [];
end
