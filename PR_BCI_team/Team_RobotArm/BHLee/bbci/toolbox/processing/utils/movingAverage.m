function xa= movingAverage(x, n, varargin)
%MOVINGAVERAGE - moving average
%
%Description:
% This function calculates the moving average of a matrix along
% the first dimension.
%
%Synopsis:
% XMA= movingAverage(X, N, 'Property1',Value1, ...);
% XMA= movingAverage(X, N, method)
% XMA= movingAverage(X, N, method, window)
%
%Arguments:
% X:      input data matrix
% N:      length of window
% method: see property 'method'
% window: see property 'window'
%
%Optional Properties:
% method: String, 'causal' or 'centered', determining how the window
%    in which the average is calculated is aligned. Default 'causal'.
% window: Can be a vector of size [N 1] that is used for weigthing the
%    average. The default scalar 1 means no weighting.
%    Can also be a function object.
% tolerate_nans: 0 or 1. If 1, the moving average is only calculated
%    over the non-NaN values of X (within the moving window). This is
%    considerable slower.
%
%Output:
% XMA:   output data matrix of the same size as input X.
%
%Example:
% x= [1:5 nan nan nan nan 6:10]';
% movingAverage([x, flipud(x)], 3, 'method','centered','tolerate_nans',1)

%Author(s): Benjamin Blankertz and an anonymous IDAheimer (weighting)


method_list= {'causal', 'centered'};

%% determine input format
if isempty(varargin) | ~ismember(varargin{1}, method_list),
  opt= propertylist2struct(varargin{:});
else
  opt= struct('method', varargin{1});
  if length(varargin)>=2,
    opt.window= varargin{2};
  end
end

opt= set_defaults(opt, ...
                  'method', 'causal', ...
                  'tolerate_nans', 0, ...
                  'window', 1);

if ~ismember(opt.method, method_list),
  error('unknown method');
end

%% input matrix x will be used as if ndims(x)=2
szx= size(x);
[T,N]= size(x);
xa= zeros(T,N);

window= opt.window;
if ischar(window) & (strcmp(window,'cos') | strcmp(window,'sin'))
  window = inline('sin(x)','x');
end

if isobject(window)
  switch(opt.method),
   case 'causal'
    window = window((1:n)'/n*pi/2);
   case 'centered'
    window = window((1:n)'/n*pi);
    %% BB: I would suggest window((1:n)'/(n+1)*pi)
  end
end

if length(window)==1 & ~opt.tolerate_nans,
  x = x*window;
  switch(opt.method),
   case 'centered',
    w0= -ceil((n-1)/2);
    w1= n-1+w0;
    xa(1,:)= mean(x(1:w1+1,:), 1);
    for k= 2:-w0(1)+1,
      xa(k,:)= (xa(k-1,:)*(w1+k-1) + x(k+w1,:))/(w1+k);
    end
    for k=-w0(1)+2:T-w1,
      xa(k,:)= xa(k-1,:) + (x(k+w1,:) - x(k+w0-1,:))/n;
    end
    for k0= 1:w1,
      k= T-w1+k0;
      xa(k,:)= (xa(k-1,:)*(n-k0+1) - x(k+w0-1,:))/(n-k0);
    end
   case 'causal',
    xa(1,:)= x(1,:);
    for k= 2:min(n,T),
      xa(k,:)= (xa(k-1,:)*(k-1) + x(k,:))/k;
    end
    for k= n+1:T,
      xa(k,:)= xa(k-1,:) + (x(k,:) - x(k-n,:))/n;
    end
  end
  
elseif opt.tolerate_nans,
  if length(window)==1,
    window= window*ones(n,1);
  end
  for cc= 1:N,
    switch(opt.method),
     case 'centered'
      ma= NaN;
      for k= 1:floor(n/2),
        iv= [1:ceil(n/2)+k-1]; 
        valid= find(~isnan(x(iv,cc)));
        if ~isempty(valid),
          wiv= [floor(n/2)+2-k:n];
          ma= sum(window(wiv(valid)).*x(iv(valid),cc)) ...
                 / sum(window(wiv(valid)));
        end
        xa(k,cc)= ma;
      end
      iv= [1:n];
      for k= floor(n/2)+1:T-ceil(n/2)+1,
        valid= find(~isnan(x(iv,cc)));
        if ~isempty(valid),
          ma= sum(window(valid).*x(iv(valid),cc))/sum(window(valid));
        end
        xa(k,cc)= ma;
        iv= iv + 1;
      end
      for k= T-ceil(n/2)+2:T,
        iv= [k-floor(n/2):T];
        valid= find(~isnan(x(iv,cc)));
        if ~isempty(valid),
          wiv= [1:T-k+floor(n/2)+1];
          ma= sum(window(wiv(valid)).*x(iv(valid),cc)) ...
              / sum(window(wiv(valid)));
        end
        xa(k,cc)= ma;
      end
      
     case 'causal',
      ma= NaN;
      for k= 1:n-1,
        iv= [1:k];
        valid= find(~isnan(x(iv,cc)));
        if ~isempty(valid),
          wiv= [n-k+1:n];
          ma= sum(window(wiv(valid)).*x(iv(valid),cc)) ...
              / sum(window(wiv(valid)));
        end
        xa(k,cc)= ma;
      end
      iv= [1:n];
      for k = n:T
        valid= find(~isnan(x(iv,cc)));
        if ~isempty(valid),
          ma= sum(window(valid).*x(iv(valid),cc))/sum(window(valid));
        end
        xa(k,cc)= ma;
        iv= iv + 1;
      end
    end
  end
  
else
  window = repmat(window, [1 N]);
  sumwin= sum(window(:,1),1);
  switch(opt.method),
   case 'centered'
    for k = 1:floor(n/2)
      xa(k,:) = sum(window(floor(n/2)+2-k:end,:).*x(1:ceil(n/2)+k-1,:),1) ...
                / sum(window(floor(n/2)+2-k:end,1),1);
    end
    for k = floor(n/2)+1:T-ceil(n/2)+1,
      xa(k,:) = sum(window.*x(k-floor(n/2):k+ceil(n/2)-1,:),1) / sumwin;
    end
    for k = T-ceil(n/2)+2:T
      xa(k,:) = sum(window(1:T-k+floor(n/2)+1,:).*x(k-floor(n/2):end,:),1) ...
                / sum(window(1:T-k+floor(n/2)+1,1),1);
    end
    
   case 'causal'
    for k = 1:n-1
      xa(k,:) = sum(window(n-k+1:end,:).*x(1:k,:),1) ...
                / sum(window(n-k+1:end,1),1);
    end
    for k = n:T
      xa(k,:) = sum(window.*x(k-n+1:k,:),1) / sumwin;
    end
    
  end
  
end

%% put results in original shape
xa= reshape(xa, szx);
