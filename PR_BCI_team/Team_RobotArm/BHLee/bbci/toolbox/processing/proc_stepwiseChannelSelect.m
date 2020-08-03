% proc_stepwiseChannelSelect - Channel selection based on the SWLDA method.
%
% Usage:
%   clab = proc_stepwiseChannelSelect(xTr, yTr, opts)
%   clab = proc_stepwiseChannelSelect(xTr, yTr, 'clab', clab)
%   [clab H] = proc_stepwiseChannelSelect(xTr, yTr, 'visualize', 1, 'clab', clab)
%
% Input:
%   xTr: Data matrix with features. Either in flatened or non flatened format. 
%   yTr: Class membership. LABELS(i,j)==1 if example j belongs to
%           class i.
%   OPTS: options structure 
%     Recognized options are:
%       'pEntry': p-value to enter regression (default: 0.1)
%       'pRemoval': variables are eliminated if p-value of partial f-test 
%             exceeds pRemoval (default: 0.15)
%       'clab': cell with channel names
%       'featPChan': in case xTr is flatened, this parameter indicates how
%             many features belong to each channel.
%       'maxChan': the maximum number of channels to be returned (default:
%             16)
%       'channelwise': if true, the features that belong to a channel will
%             alway enter/leave the selection in a group. Enter criteria is
%             evaluated on the lowest p-value of remaining features. For the
%             leave criterion, the lowest p-value of each channel is
%             compared to the threshold. 
%             If false, the procedure functions on a feature basis. Once
%             features are selected that belong to maxChan channels, the
%             procedure is stopped.
%       'visualize': if true, plots the channels on a scalp.
%
% Output:
%   clab: contains channel names of selected channels if clab is set.
%             Otherwise it contains their indices.
%   H: if visualize is set, this contains the handle to the created figure.
%
% Description:
%   proc_stepwiseChannelSelect selects a number of discriminative channels
%   following the stepwise part of the SWLDA. The stepwise procedure stops 
%   when there are no more variables falling below the critical p-value of
%   pEntry. It can be limited by maxChan, the maximum number of channels
%   that should be selected. The critical p-value for removal is
%   set to pRemoval.
%
%
%   References: N.R. Draper, H. Smith, Applied Regression Analysis, 
%   2nd Edition, John Wiley and Sons, 1981. This function implements
%   the algorithm given in chapter 'Computional Method for Stepwise 
%   Regression' of the first edition (1966).
%   
%   
%   See also TRAIN_SWLDA,
%   
%   Martijn Schreuder, 2010

function varargout = proc_stepwiseChannelSelect(xTr, yTr, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
        'pEntry', 0.1, ...
        'pRemoval', .15, ...
        'clab', {}, ...
        'featPChan', [], ...
        'maxChan', 16, ...
        'channelwise', 1, ...
        'visualize', 1);

if ndims(xTr) == 3,
    opt.featPChan = size(xTr,1);
    xTr = squeeze(reshape(xTr, 1, [], size(xTr,3)));
elseif isempty(opt.featPChan),
    error('In case of a flatened xTr, opt.featPChan needs to be set');
end

if opt.visualize && isempty(opt.clab),
    warning('Please give clab if you want visualization. visualize set to False');
    opt.visualize = 0;
end
  
if size(yTr,1) == 1, yTr = [yTr<0; yTr>0]; end
ind = find(sum(abs(xTr),1)==inf);
xTr(:,ind) = [];
yTr(:,ind) = [];

x=xTr;

xr = x';
y = yTr(1,:)';
subset = [];

dat = [xr y];

n = size(dat,1);   % number of trials
k = size(dat,2);   % number of independent variables + 1 (response)
C.w = zeros(k-1,1);
SSu = sum(dat,1);  % uncorrected sums of squares
SCu = dat'*dat;    % uncorrected cross-products

% corrected cross-products
SCc = SCu - SSu'*SSu./n;
s2 = diag(SCc);

% correlation coefficients
R = SCc ./ sqrt(s2*s2');

A = [R [eye(k-1); zeros(1,k-1)]; [-1.*eye(k-1) zeros(k-1,1) zeros(k-1,k-1)]];

chn = zeros(opt.featPChan, (k-1)/opt.featPChan);
conv_mat = reshape([1:k-1], opt.featPChan, []);

dorun=true; 

while((length(find(sum(chn))) < opt.maxChan) & dorun)
  
  % statistics
  for i=1:k-1
    V(i) = A(i,k)*A(k,i)/A(i,i);
  end

  fi = find(V==max(max(V)));

  if opt.channelwise,
      [row add_id] = find(conv_mat == fi);
  else
      add_id = fi;
  end
  
  % best variable to enter regression
  dfres = n - 2 - size(subset,2);

  % MSreg = A(fi,k)^2;
  % MSres = (A(k,k)^2 - A(fi,k)^2) / dfres;

  F = dfres*V(fi) / (A(k,k)-V(fi));
  p = ones(size(F,1),size(F,2)) - proc_fcdf(F,size(subset,2)+1,dfres);

  if(p < opt.pEntry) 
    if opt.channelwise,
        to_add = conv_mat(:,add_id);
    else
        to_add = add_id;
    end
    subset = union(subset,add_id); 
    
    for i = 1:length(to_add),
        B = A -  A(:,to_add(i))*A(to_add(i),:)./A(to_add(i),to_add(i)); 
        B(to_add(i),:) = A(to_add(i),:)/A(to_add(i),to_add(i));
        A = B;
    end

  else
    B = A;
    dorun = false;
  end

  % test for elimination of variables already in regression
  if(size(subset,2)>1),
    old = setdiff(subset,add_id);
    for i=old
      if opt.channelwise,
          to_rem = conv_mat(:,i)';
      else
          to_rem = i;
      end
      
      for j = 1:length(to_rem)
          Fp(j) = dfres*(B(to_rem(j),k)^2) / (B(k,k)*B(k+to_rem(j),k+to_rem(j)));
          pp(j) = ones(size(Fp(j),1),size(Fp(j),2)) - proc_fcdf(Fp(j),1,dfres);
      end

      if(min(pp) > opt.pRemoval)   % eliminate
        subset = setdiff(subset,i);
        
        for j = 1:length(to_rem),
            % adapt matrices after elimination
            A = B -  B(:,k+to_rem(j))*B(k+to_rem(j),:)./B(k+to_rem(j),k+to_rem(j));
            A(to_rem(j),:) = B(to_rem(j),:)/B(k+to_rem(j),k+to_rem(j));
            B = A;
        end
      end
    end
  end
  A = B;
  chn = zeros(opt.featPChan, (k-1)/opt.featPChan);
  if opt.channelwise,
      chn(:,subset) = 1;
  else
      chn(subset) = 1;
  end

end % while ~stoppingrule

% add all features for the selected channels
chn(:,find(sum(chn))) = 1;
subset = find(chn)';

if nargout > 0,
    varargout{1} = find(sum(chn));
    if ~isempty(opt.clab),
        varargout{1} = opt.clab(varargout{1});
    end
end   
H = [];
if opt.visualize,
    H = drawElectrodesOnSchalp(opt.clab, varargout{1});
end
if nargout > 1,
    varargout{2} = H;
end

end 

function H = drawElectrodesOnSchalp(clab, highlight),
    mnt = getElectrodePositions(clab)
    colOrder = [0.9 0 0.9; 0.4 0.57 1];
    labelProps = {'FontName','Times','FontSize',8,'FontWeight','normal'};
    markerProps = {'MarkerSize', 15, 'MarkerEdgeColor','k','MarkerFaceColor',[1 1 1]};
    highlightProps = {'MarkerEdgeColor','k','MarkerFaceColor',colOrder(1,:),...
        'LineWidth',2};
    linespec = {'Color' 'k' 'LineWidth' 2};
    refProps = {'FontSize', 8, 'FontName', 'Times','BackgroundColor',[.8 .8 .8],'HorizontalAlignment','center','Margin',2};

    opt = {'showLabels',1,'labelProps',labelProps,'markerProps',...
        markerProps,'markChans',highlight,'markMarkerProps',...
        highlightProps,'linespec',linespec,'ears',1,'reference','nose', ...
        'referenceProps', refProps};

    % Draw the stuff
    H= drawScalpOutline(mnt, opt{:});
    H.fig = gcf;
    set(H.fig, 'MenuBar', 'none');
    set(gca,'box','on')
    set(gca, 'Position', [0 0 1 1]);
    pos = get(gcf,'Position');
    axis off;
end
