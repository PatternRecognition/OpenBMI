function [efs,F,cdfs,p,eps,dfs,b,y2,sig]=repanova(d,D,fn,gg,alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R.Henson, 17/3/03; rik.henson@mrc-cbu.cam.ac.uk
%
% General N-way (OxPxQxR...) repeated measures ANOVAs (no nonrepeated factors)
%
% Input:
%
% d = data	A matrix with rows = replications (eg subjects) and
%		              columns = conditions 
%
% D = factors	A vector with as many entries as factors, each entry being
%		the number of levels for that factor
%
%		Data matrix d must have as many columns (conditions) as
%		the product of the elements of the factor matrix D
%
%		First factor rotates slowest; last factor fastest
% 
% 	Eg, in a D=[2 3] design: factor A with 2 levels; factor B with 3:
%	    data matrix d must be organised:
%
%		A1B1	A1B2	A1B3	A2B1	A2B2	A2B3
% 	rep1
%	rep2
%	...
%	
% Output:
%
% efs 	= effect, eg [1 2] = interaction between factor 1 and factor 2
% F   	= F value
% cdfs 	= corrected df's (using Greenhouse-Geisser)
% p     = p-value
% eps   = epsilon
% dfs   = original dfs
% b     = betas
% y2    = cell array of means for each level in a specific ANOVA effect
% sig   = cell array of significant effects (uncorrected and Bonferroni corrected)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<5
	alpha=0.05;
end

if nargin<4
	gg=1;		% G-G correction
end

if nargin<3		% No naming of factors provided
   for f=1:length(D)
	fn{f}=sprintf('%d',f);
   end
end

Nf = length(D);		% Number of factors
Nd = prod(D);		% Number of conditions
Ne = 2^Nf - 1;		% Number of effects
Nr = size(d,1);		% Number of replications (eg subjects)

sig=cell(2,1);

if size(d,2) ~= Nd
	error(sprintf('data has %d conditions; design only %d',size(d,2),Nd))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sc = cell(Nf,2);	% create main effect/interaction component contrasts
for f = 1 : Nf
	sc{f,1} = ones(D(f),1);
	sc{f,2} = detrend(eye(D(f)),0);
end 

sy = cell(Nf,2);	% create main effect/interaction components for means
for f = 1 : Nf
	sy{f,1} = ones(D(f),1)/D(f);
	sy{f,2} = eye(D(f));
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for e = 1 : Ne		% Go through each effect

	cw = num2binvec(e,Nf)+1;

	c  = sc{1,cw(Nf)};	% create full contrasts
	for f = 2 : Nf
		c = kron(c,sc{f,cw(Nf-f+1)});
	end

	y = d * c;		% project data to contrast sub-space

	cy  = sy{1,cw(Nf)};	%  calculate component means
	for f = 2 : Nf
		cy = kron(cy,sy{f,cw(Nf-f+1)});
	end
	y2{e} = mean(d * cy);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	nc = size(y,2);
	df1 = rank(c);
	df2 = df1*(Nr-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% long GLM way (no GG):
%
%	y = y(:);
%	X = kron(eye(nc),ones(Nr,1));
%	b{e} = pinv(X)*y;
%	Y = X*b{e};
%	R = eye(nc*Nr)- X*pinv(X);
%	r = y - Y;
%%	V = r*r';
%	V = y*y';
%	eps(e) = trace(R*V)^2 / (df1 * trace((R*V)*(R*V)));
%
%%	ss = Y'*y;
%%	mse = (y'*y - ss)/df2;
%%	mss = ss/df1;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computationally simpler way

	b{e} = mean(y);			
	ss   =  sum(y*b{e}');
	mse  = (sum(diag(y'*y)) - ss)/df2;
	mss  =  ss/df1;

	if gg
		V      = cov(y);			% sample covariance
		eps(e) = trace(V)^2 / (df1*trace(V'*V));% Greenhouse-Geisser 
	else
		eps(e) = 1;
	end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	efs{e} = Nf+1-find(cw==2);			% codes which effect 

	F(e)   = mss/mse;

	dfs(e,:)  = [df1 df2];
	cdfs(e,:) = eps(e)*dfs(e,:);

	p(e) = 1-spm_Fcdf(F(e),cdfs(e,:));

	if p(e) < alpha; sig{1}=[sig{1} e]; end
	if p(e) < alpha/Ne; sig{2}=[sig{2} e]; end

	en=fn{efs{e}(1)};	% Naming of factors
	for f = 2:length(efs{e})
		en = [fn{efs{e}(f)} en];
	end

	disp(sprintf('Effect %02d: %-18s F(%3.2f,%3.2f)=%4.3f,\tp=%4.3f',...
		e,en,cdfs(e,1),cdfs(e,2),F(e),p(e)))
end

disp(sprintf('\n\nsula special\n'))

for i=1:numel(y2)
fprintf('y2[%d] = \n',i);
disp(y2{i})
end

for i=1:numel(sig)
fprintf('sig[%d] = \n',i);
disp(sig{i})
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sub-function to code all main effects/interactions

function b = num2binvec(d,p)

if nargin<2
	p = 0;		% p is left-padding with zeros option
end

d=abs(round(d));

if(d==0)
	b = 0;
else
	b=[];
 	while d>0
		b=[rem(d,2) b];
		d=floor(d/2);
 	end
end

b=[zeros(1,p-length(b)) b];

