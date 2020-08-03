function xround = roundto(x, d)
if d<= 0, error('Error: Rounding precision must be > 0'); end
xround = round(x/d)*d;