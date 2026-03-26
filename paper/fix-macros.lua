-- TODO: remove once using rjtools output format
function RawInline(el)
  if el.format == "tex" or el.format == "latex" then
    local pkg = el.text:match("\\CRANpkg{(.-)}")
    if pkg then
      return pandoc.Strong(pandoc.Str(pkg))
    end
  end
end
