function mexp = ameanexp(f, UC, mexp); 
% ameanexp adaptive mean.
% function mexp = ameanexp(f, UC, mexp0); 
%   f 	input data  
%   UC 	update coefficient 0 < UC << 1
%   mexp0 initial mean
%   mexp estimated mean

[r,p] = size(f);

if nargin< 2,
        fprintf(2,'Error: missing update coefficient\n');
        return;
else	
        if ~((UC > 0) & (UC <1)),
                fprintf(2,'Error : update coefficient not within range [0,1]\n');
                return;
        end;
        if UC > 1/p,
                fprintf(2,'Warning: update coefficient should be smaller than 1/number_of_dimensions\n');
        end;
end;

if nargin<3,
        mexp = [];
end;

if isempty(mexp),
        mexp = zeros(1,p+Mode_E);
end;

for k = 1:r,
                mexp = (1-UC)*mexp + UC*f(k,:);
end;