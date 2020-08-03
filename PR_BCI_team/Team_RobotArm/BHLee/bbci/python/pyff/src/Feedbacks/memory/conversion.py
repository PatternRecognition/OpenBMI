'''
Created on 24.06.2009

@author: sophie
'''
"""Conversion functions between various Chinese encodings and representations.

The following terms are used to represent the encodings / representation used
in the conversion functions (the samples on the right are for the character
U+4E00 (yi1; "one")):

GB2312 (Kuten/Quwei form): "5027" [used in the "GB2312" field of Unihan.txt]
GB2312 (ISO-2022 form):    "523B" [the "internal representation" of GB code]
EUC-CN:                    "D2BB" [this is the "external encoding" of GB2312-
                                   ISO2022's "internal representation"; also
                                   the form that Ocrat uses]
UTF-8:                     "E4 B8 80" [used in the "UTF-8" field in Unihan.txt]
-------------------------------------------------------------------------------
Unihan UCN:                "U+4E00"   [used by Unicode Inc.]
Unihan NCR (decimal):      "&#19968;" [Numerical Character Reference ...
Unihan NCR (hex):          "&x4E00;"   ... used in XML/HTML/SGML]
-------------------------------------------------------------------------------
internal Python unicode:   u"\u4e00"  [this is the most useful form!]
internal Python 'utf8':    "\\xe4\\xb8\\x80"
internal Python 'gb2312':  "\\xd2\\xbb"
internal Python 'euc-cn':  "\\xd2\\xbb"
internal Python 'gb18030': "\\xd2\\xbb"

See these resources for more information:
 * Wikipedia "Extended_Unix_Code" article
   * "EUC-CN is the usual way to use the GB2312 standard for simplified Chinese
     characters ... the ISO-2022 form of GB2312 is not normally used"
 * Wikipedia "HZ_(encoding)" article (the example conversion)
 * Wikipedia "Numeric_character_reference" article
 * Unihan (look for "Encoding forms", "Mappings to Major Standards")
   * e.g. http://www.unicode.org/cgi-bin/GetUnihanData.pl?codepoint=4E00
"""

import re

hexd = lambda s: hex(s)[2:] #only the hex digits (strip '0x' at the beginning)

def kuten_to_gb2312(kuten):
    """Convert GB kuten / quwei form (94 zones * 94 points) to GB2312-1980 / ISO-2022-CN hex (internal representation)"""
    zone, point = int(kuten[:2]), int(kuten[2:])
    hi, lo = hexd(zone + 0x20), hexd(point + 0x20)
    return "%s%s" % (hi, lo)

def gb2312_to_euc(gb2312hex):
    """Convert GB2312-1980 hex (internal representation) to EUC-CN hex (the "external encoding")"""
    hi,lo = int(gb2312hex[:2], 16), int(gb2312hex[2:], 16)
    hi,lo = hexd(hi + 0x80), hexd(lo + 0x80)
    return "%s%s" % (hi, lo)

def euc_to_utf8(euchex):
    """Convert EUC hex (e.g. "d2bb") to UTF8 hex (e.g. "e4 b8 80")"""
    utf8 = euc_to_python(euchex).encode("utf-8")
    utf8 = repr(utf8)[1:-1].replace("\\x", " ").strip()
    return utf8

#TODO: make these conversions go both ways (i.e. gb2312_to_kuten, euc_to_gb2312, utf8_to_euc)

#TODO: fill in the conversions? (i.e. create kuten_to_utf8, so you don't have to compose kuten_to_gb2312 and gb2312_to_euc by hand)
#TODO: might even want to create a more generic system so that you can just pick the encoding types and dispatch the right method

#TODO: more encoding formats? (HZ at least, it's easy) ; (actually HZ is an included in the codecs.)

## convert to internal Python unicode / string objects ########################

def ucn_to_python(ucn):
    """Convert a Unicode Universal Character Number (e.g. "U+4E00" or "4E00") to Python unicode (u'\\u4e00')"""
    ucn = ucn.strip("U+")
    if len(ucn) > 4:
        return eval("u'\U%08x'" % int(ucn, 16)) #this would be a security hole if we didn't use `int` to make the input safe
    else:
        return eval("u'\u%s'" % ucn) #4 characters isn't enough room to do damage with the eval
        #TODO: this dies on decimal input (e.g. ucn_to_python("100")

def euc_to_python(hexstr):
    """Convert a EUC-CN (GB2312) hex to a Python unicode string"""
    hi = hexstr[0:2]
    lo = hexstr[2:4]
    gb_enc = eval("'\\x%s\\x%s'" % (hi,lo)) #hi and lo are only 2 characters long, no risk with eval-ing them
    return gb_enc.decode("gb2312")

def ncr_to_python(ncr):
    """Convert Unicode Numerical Character Reference (e.g. "19968", "&#19968;", or "&#x4E00;") to native Python Unicode (u'\\u4e00')"""
    ncr = ncr.lower()
    if ncr.startswith("&#x"): 
        #hex NCR
        ncr = ncr.strip("&#x;")
        ncr = ncr.swapcase()
    elif ncr.startswith("&#"):
        #decimal NCR
        ncr = ncr.strip("&#;")
        ncr = hexd(int(ncr))
    else:
        #assume it's a decimal NCR not in the XML character entity ref. format
        ncr = hexd(int(ncr))
    return ucn_to_python(ncr)

## convert from internal Python unicode / string objects ######################

def python_to_ucn(uni_char):
    """Converts a one character Python unicode string (e.g. u'\\u4e00') to the corresponding Unicode UCN ('U+4E00')"""
    ucn = repr(uni_char)[4:-1]
    if len(ucn) > 4:
       ucn = ucn.lstrip("0") #get rid of the zeroes that Python uses to pad 32 byte UCNs
    return "U+%s" % ucn

def python_to_euc(uni_char):
    """Converts a one character Python unicode string (e.g. u'\\u4e00') to the corresponding EUC hex ('d2bb')"""
    return repr(uni_char.encode("gb2312"))[1:-1].replace("\\x", "")

def python_to_ncr(uni_char, **options):
    """Converts a one character Python unicode string (e.g. u'\\u4e00') to the corresponding Unicode NCR ('&x4E00;')
    Change the output format by passing the following parameters:
     * no parameters - default behavior: hex=True, xml=True
     * decimal=True - output the decimal value instead of hex
     * hex=False - (same as decimal=True)
     * xml=False - just display the decimal or hex value, i.e. strip off the '&#', '&x', and ';'"""
    hexflag, decflag, xmlflag = map(options.get, ["hex", "decimal", "xml"], [True, False, True])
    if decflag:
        out = int(repr(uni_char)[4:-1], 16)
        if xmlflag:
            out = "&#%s;" % out
    elif hexflag:
        out = repr(uni_char)[4:-1]
        if xmlflag:
            out = "&#x%s;" % out
    return out

## handle multiple characters #################################################

def string_to_ncr(uni_string, **options):
    """Converts a Python unicode string (e.g. u'p\\u012bn y\\u012bn') to the corresponding Unicode NCRs.
    See `python_to_ncr` for formatting options."""
    for char in uni_string:
        if ord(char) > 128:
            uni_string = uni_string.replace(char, python_to_ncr(char, options=options))
    return uni_string

def ncrstring_to_python(ncr_string):
    """Convert a string of Unicode NCRs (e.g. "&#19968;&#x4E00;") to native Python Unicode (u'\\u4e00\\u4e00')"""
    res = re.findall("&#[x0-9a-fA-F]*?;", ncr_string)
    for r in res:
        ncr_string = ncr_string.replace(r, ncr_to_python(r))
    return ncr_string
