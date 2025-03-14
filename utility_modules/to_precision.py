"""
Source: https://github.com/BebeSparkelSparkel/to-precision/releases/tag/0.0.0
Source was referenced in: http://randlet.com/blog/python-significant-figures-format/
The code in this module is from the GitHub site.
Style modification applied by C.S.Echt.
"""

__author__ = """William Rusnack github.com/BebeSparkelSparkel linkedin.com/in/williamrusnack williamrusnack@gmail.com
Eric Moyer github.com/epmoyer eric@lemoncrab.com"""

from math import floor, log10

# Note: Type hints for function parameters are as follows:
#   value: Union[int, float, str]
#   precision: int
#   notation: str
#   delimiter str
#   auto_limit: int,
#   strip_zeros: bool
#   preserve_integer: bool

def std_notation(value, precision) -> str:
    """
    Standard notation (US version).
    ref: http://www.mathsisfun.com/definitions/standard-notation.html

    ex:
    std_notation(5, 2) => '5.0'
    std_notation(5.36, 2) => '5.4'
    std_notation(5360, 2) => '5400'
    std_notation(0.05363, 3) => '0.0536'

    created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Args:
        value: any type that can be converted to a float
        precision: integer that is greater than zero
    Returns:
        Tuple of Args to be converted to the specified *precision* in
        standard notation.
    """
    return to_precision(value, precision, notation='std')


def sci_notation(value, precision, delimiter='e') -> str:
    """
    Scientific notation with exponentiation.
    ref: https://www.mathsisfun.com/numbers/scientific-notation.html

    ex:
    sci_notation(123, 1, 'E') => '1E2'
    sci_notation(123, 3, 'E') => '1.23E2'
    sci_notation(.126, 2, 'E') => '1.3E-1'

    created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Args:
      value: any type that can be converted to a float
      precision: integer that is greater than zero
      delimiter: is placed between the decimal value and 10s exponent

    Returns:
      Tuple of Args to be converted to the specified *precision* in
      scientific notation with 10s exponent *delimiter* placed between
      the decimal value and 10s exponent.
    """
    return to_precision(value, precision, notation='sci', delimiter=delimiter)


def eng_notation(value, precision, delimiter='e') -> str:
    """
    Engineering notation with exponentiation
    ref: http://www.mathsisfun.com/definitions/engineering-notation.html

    ex:
    sci_notation(123, 1, 'E') => '100E0'
    sci_notation(1230, 3, 'E') => '1.23E3'
    sci_notation(.126, 2, 'E') => '120E-3'

    created by William Rusnack
    github.com/BebeSparkelSparkel
    linkedin.com/in/williamrusnack/
    williamrusnack@gmail.com

    Args:
      value: any type that can be converted to a float
      precision: integer that is greater than zero
      delimiter: is placed between the decimal value and 10s exponent

    Returns:
        Tuple of Args to be converted to the specified *precision* in
        scientific notation with 10s exponent that is divisible by 3 with
        *delimiter* placed between the decimal value and 10s exponent.
    """
    return to_precision(value, precision, notation='eng', delimiter=delimiter)


def auto_notation(value, precision, delimiter='e') -> str:
    """
    Automatically selects between standard notation (US version) and
    scientific notation. Values in the range 0.001 < abs(value) < 1000
    return standard notation.

    http://www.mathsisfun.com/definitions/standard-notation.html
    https://www.mathsisfun.com/numbers/scientific-notation.html

    ex:
    auto_notation(123, 4) => '123.0'
    std_notation(1234, 4) => '1.234e3'

    Args:
      value: any type that can be converted to a float
      precision: integer that is greater than zero
      delimiter: is placed between the decimal value and 10s exponent

    Returns:
      Tuple of Args to be converted to the specified *precision* in
      the proper notation.
    """

    return to_precision(value, precision, notation='auto', delimiter=delimiter)


def to_precision(value,
                 precision: int,
                 notation='auto',
                 delimiter='e',
                 auto_limit=4,
                 strip_zeros=False,
                 preserve_integer=False) -> str:
    """
    Converts a value to the specified notation and precision.

    Args:
      value: any type that can be converted to a float
      precision: integer that is greater than zero
      notation: string
        'auto' - selects standard notation when abs(power) < auto_limit else
          returns scientific notation.
        'sci' or 'scientific' - returns scientific notation
          ref: https://www.mathsisfun.com/numbers/scientific-notation.html
        'eng' or 'engineering' - returns engineering notation
          ref: http://www.mathsisfun.com/definitions/engineering-notation.html
        'std' or 'standard' - returns standard notation
          ref: http://www.mathsisfun.com/definitions/standard-notation.html
      delimiter: is placed between the decimal value and 10s exponent
      auto_limit: integer. When abs(power) exceeds this limit, 'auto'
        mode will return scientific notation.
      strip_zeros: if true, trailing decimal zeros will be removed.
      preserve_integer: if true, 'std' will preserve all digits when returning
        values that have no decimal component.

    Returns:
      Tuple of Args to be converted to the specified *notation*.
    """

    is_neg, sig_digits, dot_power, ten_power = _sci_decompose(value, precision)

    converter = {
        'auto': _std_notation if abs(ten_power) < auto_limit else _sci_notation,
        'sci': _sci_notation,
        'scientific': _sci_notation,
        'eng': _eng_notation,
        'engineering': _eng_notation,
        'std': _std_notation,
        'standard': _std_notation
    }.get(notation)

    if converter is None:
        raise ValueError('Unknown notation: ' + str(notation))

    return converter(value, precision, delimiter, strip_zeros, preserve_integer)

def _std_notation(value, precision, _, strip_zeros, preserve_integer) -> str:
    """
    Standard notation (US version).
    ref: http://www.mathsisfun.com/definitions/standard-notation.html

    Args:
        value: any type that can be converted to a float
        precision: integer that is greater than zero
        strip_zeros: if true, trailing decimal zeros will be removed.
        preserve_integer: if true, 'std' will preserve all digits when
            returning values that have no decimal component.

    Returns:
        A string of value with the proper precision.
    """
    sig_digits, power, is_neg = _number_profile(value, precision)
    result = ('-' if is_neg else '') + _place_dot(sig_digits, power, strip_zeros)

    if preserve_integer and not '.' in result:
        # Result was an integer, preserve all digits
        result = f'{value:0.0f}'

    return result


def _sci_notation(value, precision, delimiter, strip_zeros, _) -> str:
    """
    Scientific notation/
    ref: https://www.mathsisfun.com/numbers/scientific-notation.html

    Args:
      value: any type that can be converted to a float
      precision: integer that is greater than zero
      delimiter: is placed between the decimal value and 10s exponent.
      strip_zeros: if true, trailing decimal zeros will be removed.

    Returns:
      Tuple of Args to be converted to the specified *precision* in
      scientific notation with 10s exponent *delimiter* placed between
      the decimal value and 10s exponent.
    """
    is_neg, sig_digits, dot_power, ten_power = _sci_decompose(value, precision)

    return ('-' if is_neg else '') + _place_dot(sig_digits, dot_power, strip_zeros) + delimiter + str(ten_power)


def _eng_notation(value, precision, delimiter, strip_zeros, _) -> str:
    """
    engineering notation
    ref: http://www.mathsisfun.com/definitions/engineering-notation.html

    returns a string of value with the proper precision and 10s exponent that is divisible by 3
    delimiter is placed between the decimal value and 10s exponent

    Args:
        value: any type that can be converted to a float
        precision: integer that is greater than zero
        delimiter: is placed between the decimal value and 10s exponent.
        strip_zeros: if true, trailing decimal zeros will be removed.

    Returns:
        Tuple of Args to be converted to the specified *precision* in
        scientific notation with 10s exponent that is divisible by 3 with
        *delimiter* placed between the decimal value and 10s exponent.
    """
    is_neg, sig_digits, dot_power, ten_power = _sci_decompose(value, precision)

    eng_power = int(3 * floor(ten_power / 3))
    eng_dot: int = dot_power + ten_power - eng_power

    return ('-' if is_neg else '') + _place_dot(sig_digits, eng_dot, strip_zeros) + delimiter + str(eng_power)


def _sci_decompose(value, precision) -> tuple:
    """
    Calculate properties for a scientific notation number used in
    sci_notation() and eng_notation().

    created by William Rusnack
    github.com/BebeSparkelSparkel
    linkedin.com/in/williamrusnack/
    williamrusnack@gmail.com

    Args:
        value: any type that can be converted to a float
        precision: integer that is greater than zero

    Returns:
        the properties to construct a scientific notation number
        used in sci_notation and eng_notation.
    """
    value = float(value)
    sig_digits, power, is_neg = _number_profile(value, precision)

    dot_power: int = -(precision - 1)
    ten_power: int = power + precision - 1

    return is_neg, sig_digits, dot_power, ten_power


def _place_dot(digits: str, power: int, strip_zeros=False) -> str:
    """
    Places the dot in the correct spot in the digits.
    if the dot is outside the range of the digits zeros will be added
    if strip_zeros is set, trailing decimal zeros will be removed

    ex:
    _place_dot('123',   2, False) => '12300'
    _place_dot('123',  -2, False) => '1.23'
    _place_dot('123',   3, False) => '0.123'
    _place_dot('123',   5, False) => '0.00123'
    _place_dot('120',   0, False) => '120.'
    _place_dot('1200', -2, False) => '12.00'
    _place_dot('1200', -2, True ) => '12'
    _place_dot('1200', -1, False) => '120.0'
    _place_dot('1200', -1, True ) => '120'

    created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Args:
        digits: The string of number values to be converted to decimal.
        power: The exponent integer to determine decimal placement.
        strip_zeros: If true, trailing decimal zeros will be removed.

    """
    if power > 0:
        out = digits + '0' * power

    elif power < 0:
        power = abs(power)
        precision = len(digits)

        if power < precision:
            out = digits[:-power] + '.' + digits[-power:]

        else:
            out = '0.' + '0' * (power - precision) + digits

    else:
        out = digits + ('.' if digits[-1] == '0' and len(digits) > 1 else '')

    if strip_zeros and '.' in out:
        out = out.rstrip('0').rstrip('.')

    return out


def _number_profile(value, precision: int) -> tuple:
    """
    Construct parameter values used by _sci_notation() and _eng_notation().

    created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Args:
        value: any type that can be converted to a float
        precision: integer that is greater than zero

    Returns:
        A tuple containing string of significant digits,
        10s exponent to get the dot to the proper location in the significant digits,
        bool that's true if value is less than zero else false

    """
    value = float(value)
    if value == 0:
        sig_digits: str = '0' * precision
        power: int = -(1 - precision)
        is_neg = False

    else:
        if value < 0:
            value = abs(value)
            is_neg = True
        else:
            is_neg = False

        power = -1 * floor(log10(value)) + precision - 1
        sig_digits = str(int(round(abs(value) * 10.0**power)))

    return sig_digits, int(-power), is_neg
