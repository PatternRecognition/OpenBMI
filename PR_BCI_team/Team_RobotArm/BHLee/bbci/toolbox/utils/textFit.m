function h= textFit(x, y, str, fit, varargin)
%h= textFit(x, y, str, fit, <textProp>)

h= text(x, y, str);
set(h, varargin{:}, 'units','normalized', 'fontUnits','normalize');

if length(fit)==1, fit= [fit inf]; end
fit(find(fit==0 | isnan(fit)))= inf;
if all(fit==inf), return; end

fontSize= min(fit(2), 2*fit(1));
set(h, 'fontSize',fontSize);
fitFontsize(h, fit);
