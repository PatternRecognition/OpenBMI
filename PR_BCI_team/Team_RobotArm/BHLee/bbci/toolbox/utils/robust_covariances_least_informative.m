function Sigma = robust_covariances_least_informative(data,epsi);

p = size(data,1);

kappa = get_kappa(p,epsi);

a = max(0,p-kappa);
b = p+kappa;
 

% START
t = mean(data,2);

dat = data-repmat(t,[1,size(data,2)]);
Sigma = dat*dat'/size(dat,2);

B = chol(Sigma); B = B'; V = inv(B);


rel = 1;

while rel>10^-3
% SCATTER
y = V*dat;

u = sum(y.*y,1);

s = min(b,max(a,u))./u;
C = (repmat(s,[size(dat,1) 1]).*y)*y'/size(dat,2);

C = C/mean(s);

B = chol(C); B = B'; W= inv(B);

V = W*V;

%LOCATION
y = V*dat;

w = min(1,sqrt(2)./sum(y.*y,1));

h = dat*w'/size(dat,2);

h = h/mean(w);

t = t+h;

dat = data-repmat(t,[1,size(data,2)]);

ww = W-eye(size(W));
ww = sum(ww(:).*ww(:));
ww = sqrt(ww);

www = V*h; www = www'*www; www = sqrt(www);
rel = max(ww,www);

end











function kappa = get_kappa(p,epsilon);

kappalow = 0;kappahigh = inf;

kappa = 1;

int = get_kappa_int(kappa,p,epsilon);

if int<1
  kappahigh = kappa;
  while int<=1
    kappa = 0.5*kappa;
    int = get_kappa_int(kappa,p,epsilon);
  end
  kappalow = kappa;
else
  kappalow = kappa;
  while int>=1
    kappa = 2*kappa;
    int = get_kappa_int(kappa,p,epsilon);
  end
  kappahigh = kappa;
end

while (kappahigh-kappalow)>=10^-10
  kappa = 0.5*(kappahigh+kappalow);
  int = get_kappa_int(kappa,p,epsilon);
  if int<1
    kappahigh = kappa;
  else
    kappalow = kappa;
  end
end






function int = get_kappa_int(kappa,p,epsilon);

a = sqrt(max(0,p-kappa));
b = sqrt(p+kappa);

i1 = (2*pi)^(-0.5*p)*exp(-0.5*a^2)*a^p/(p-a^2);
i2 = (2*pi)^(-0.5*p)*exp(-0.5*b^2)*b^p/(b^2-p);

p = p-1;

p0 = mod(p,2);
if p0==0
  sta = (erf(b/sqrt(2))-erf(a/sqrt(2)))*sqrt(0.5*pi);
else
  sta = exp(-0.5*a^2)-exp(-0.5*b^2);
end

while p0<p
  sta = sta*(p0+1)+exp(-0.5*a^2)*a^(p0+1)-exp(-0.5*b^2)*b^(p0+1);
  p0 = p0+2;
end

p = p+1;

sta = sta*(2*pi)^(-0.5*p);

int = i1+i2+sta;
int = int*(1-epsilon)*2*pi^(0.5*p)/gamma(0.5*p);

