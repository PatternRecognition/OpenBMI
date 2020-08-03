function d = orthogonal_design(nRows,nLevels)

% ORTHOGONAL_DESIGN - help function to create matrices with factor values
% assuming an orthogonal design (ie, there is data for each possible
% pairing of subconditions).
% Example: If you have the factors target/nontarget, and electrode 
% (Fz, Cz, Pz), and you measure the effect of these variables on 12
% subjects, you would get a 12 x 6 (=2*3) matrix wherein the rows represent 
% subjects, and the columns are:
% (1)target-Fz (2)nontarget-Fz (3)target-Cz (4)nontarget-Cz (5)target-Pz (6)nontarget-Pz
% The respective function call would be orthogonal_design(12,[2 3])
%
% Synopsis:
%   D = ORTHOGONAL_DESIGN(NROWS,NLEVELS)
%
% Arguments:
%   NROWS  : number of measurements of each factor (eg number of subjects)
%   NLEVELS: a vector with each field specifying the number of levels of
%     each factor
%
% Returns:
%   A struct with the following fields
%   .mat: a cell array with the corresponding matrices for each variable
%   (except the first)
%   .anova: vector notation of the factor matrices which you can feed
%   directly into anovan()

%  2009 Matthias Treder

nFactors = numel(nLevels);
nCols = prod(nLevels);
d = strukt('mat',zeros(nRows,nCols,nFactors),'anova',[],'anova_mat',[]);

% Construct first row and then copy n times
for ii=1:nFactors
  repLev = nCols / prod(nLevels(ii:end)); % Number of repeats of a levels

  if ii<nFactors   % Number of repeats of the sequence
    repSeq = prod(nLevels(ii+1:end));
  else repSeq=1;
  end

  row = [];
  for rr=1:repSeq
    for nn=1:nLevels(ii)
      row = [row repmat(nn,[1 repLev])];
    end
  end
  d.mat(1,:,ii)=row;
  %% Extend to all rows
  d.mat(:,:,ii) = repmat(row,[nRows 1]);
end

for ii=1:nFactors
  dd = d.mat(:,:,ii);
  d.anova{ii} = dd(:);
  d.anova_mat = [d.anova_mat dd(:)];
end

  
  
  
  
  
  
  
  
  
  
  