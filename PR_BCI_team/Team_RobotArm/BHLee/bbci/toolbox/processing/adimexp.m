function IR = adimexp(f, UC, IR, isextended); 
% ADIM adaptive information matrix. Estimates the inverse
%   correlation matrix in an adaptive way. 
%
%   IR = adimexp(f, UC, IR0); 
%   f 	input data  (without the mean if not extended)
%   UC 	update coefficient 0 < UC << 1
%   IR0	initial information matrix
%   IR	information matrix (inverse correlation matrix)
%   the features should have the mean removed if non extended
% 	if not extended, f should be without the mean value.
%   ver como decir si extendida o no extendida.

[ur,p] = size(f);

Mode_E = 1;

if all(f(:,1)==1)
    Mode_E = 0;
end;

if ~isextended
	Mode_E=0;
	if all(f(:,1)==1)
		f=f(:,2:end);
	end;
end;

if nargin<2,
        fprintf(2,'Error ADIM: missing update coefficient\n');
        return;
else	
        if ~((UC > 0) & (UC <1)),
                fprintf(2,'Error ADIM: update coefficient not within range [0,1]\n');
                return;
        end;
        
        if UC > 1/p,
        fprintf(2,'Warning ADIM: update coefficient should be smaller than 1/number_of_dimensions\n');
        end;
end;



if nargin<3,
        IR = [];
end;

if isempty(IR), 
    IR = eye(p+Mode_E);
end;
	D = zeros(ur,(p+Mode_E)^2);
	W = eye(p+Mode_E)*UC/(p+Mode_E);
	W2= eye(p+Mode_E)*UC*UC/(p+Mode_E);

for k = 1:ur,
        if ~Mode_E,
                % data have already the mean
                d  = f(k,:);	
        else
                % add mean to the data...
                d  = [1 f(k,:)];
        end;
       
        if ~any(isnan(d)),
            v  = IR*d';
            IR =(1/(1-UC))*(0.5*(IR+IR') - (UC/(1-UC+UC*d*v))*v*v');
        end;    
end;