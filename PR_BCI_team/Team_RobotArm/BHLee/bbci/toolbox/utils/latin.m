function lat = latin(num)
% TRANSLATE INTEGERS NUM TO LATIN NUMBERS LAT
%
% usage:
%  lat = latin(num);
%
% input:
%  num:  a integer array
% 
% output:
%  lat:  a cell array of latins
%
% Guido Dornhege, 27/04/2004

let = {'I','V','X','L','C','D','M'};

lat = cell(1,length(num));

for i = 1:length(num)
  nu = num(i);
  if nu<=0 | nu>=3000, lat{i} = 'nan';else
    nu1 = mod(nu,10);
    str= {'','I','II','III','IV','V','VI','VII','VIII','IX'};
    lat{i} = str{nu1+1};
    
    nu = (nu-nu1)/10;
    nu1 = mod(nu,10);
    str = {'','X','XX','XXX','XL','L','LX','LXX','LXXX','XC'};
    lat{i} = [str{nu1+1},lat{i}];
    
    nu = (nu-nu1)/10;
    nu1 = mod(nu,10);
    str = {'','C','CC','CCC','CD','D','DC','DCC','DCCC','CM'};
    lat{i} = [str{nu1+1},lat{i}];
   
    nu = (nu-nu1)/10;
    nu1 = mod(nu,10);
    str = {'','M','MM','MMM'};
    lat{i} = [str{nu1+1},lat{i}];
   
  end
    
    
    
end