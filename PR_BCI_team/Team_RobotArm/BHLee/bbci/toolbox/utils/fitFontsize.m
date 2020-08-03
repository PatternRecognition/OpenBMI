function fitFontsize(h, fit)
%fitFontsize(h, fit)

fontSize= get(h, 'fontSize');
ext= get(h, 'extent');
if fit(1)<inf,
  while ext(3)>fit(1),
    fontSize= fontSize*fit(1)/ext(3);
    set(h, 'fontSize', fontSize);
    ext= get(h, 'extent');
  end
end
if fit(2)<inf,
  while ext(4)>fit(2),
    fontSize= fontSize*fit(2)/ext(4);
    set(h, 'fontSize', fontSize);
    ext= get(h, 'extent');
  end
end
set(h, 'FontUnit','normalized');
