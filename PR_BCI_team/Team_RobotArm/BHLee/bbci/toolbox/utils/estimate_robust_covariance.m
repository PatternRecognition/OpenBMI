function Sigma = estimate_robust_covariance(data,typ);

if ~exist('typ','var') | isempty(typ)
  typ = 'norm_1';
end

if strcmp(typ,'minvol')
  Sigma = minvol(data);
  Sigma = Sigma.robustCov;
else
  val = feval(typ,data);
  
  Sigma = zeros(size(data,1));
  
  for i = 1:size(data,1);
    Sigma(i,i) = val(i)^2;
    for j = i+1:size(data,1);
      Sigma(i,j) = 0.25*(feval(typ,data(i,:)/val(i)+data(j,:)/val(j))^2-feval(typ,data(i,:)/val(i)-data(j,:)/val(j))^2)*val(i)*val(j);
      Sigma(j,i)= Sigma(i,j);
    end
  end
end
  






function val = norm_1(x);

val = sum(abs(x-repmat(mean(x,2),[1,size(x,2)])),2)/size(x,2);


function val = norm_2(x);

val = std(x,1,2);


function val = mad(x);

x = x-repmat(median(x,2),[1,size(x,2)]);

x = abs(x);

val = 1.4828*median(x,2);

