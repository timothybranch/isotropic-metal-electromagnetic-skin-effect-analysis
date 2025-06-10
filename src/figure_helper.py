import os
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cycler import cycler
from numpy.typing import NDArray
from scipy.constants import mu_0, pi

## Not needed, and gives a syntax error, but it works
tab10 = plt.get_cmap('tab10').colors
tab20 = plt.get_cmap('tab20').colors
# color2_default = plt.get_cmap('tab20').colors[0]
color2_default = plt.get_cmap('tab20')(0)

## Define some common colours
colour_AgAu = plt.get_cmap('tab10')(7)
qmi_blue = (12/255, 35/255, 68/255)
qmi_red = (186/255, 0/255, 8/255)


def set_plot_template(template_name='2ColWide', legend_outline=False):
    """
    Close all open figures, then set the plot template to use.

    Additionally, set the colour cycle to 'tab10', image colormap to 'viridis', and the figure dpi to 300.

    Parameters
    ----------
    template_name : str
        The name of the plot template to use.
    
    Returns
    -------
    None
    """

    plt.close('all')
    plt.clf()

    template_filename = 'FigureStyle-' + template_name + '.mplstyle'
    template_filepath = os.path.join(os.path.dirname(__file__), '..', 'Figures', template_filename)

    try:
        plt.style.use(template_filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f'Template file not found: {template_filepath}')

    # if template_name == '1ColWide':
    #     plt.style.use('../Figures/FigureStyle-1ColWide.mplstyle')
    # elif template_name == '2ColWide':
    #     plt.style.use('../Figures/FigureStyle-2ColWide.mplstyle')
    # elif template_name == '1ColTall':
    #     plt.style.use('../Figures/FigureStyle-1ColTall.mplstyle')
    # elif template_name == '2ColTall':
    #     plt.style.use('../Figures/FigureStyle-2ColTall.mplstyle')
    # else:
    #     raise ValueError('Invalid template name')

    ## Make the font sans-serif
    # matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'

    ## Set the colour cycle to 'tab10' and the image colormap to 'viridis'
    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=plt.get_cmap('tab10').colors) ## I can't get this to work in the style files
    # matplotlib.rcParams['image.cmap'] = 'viridis'
    # plt.rcParams['figure.dpi'] = 300

    if legend_outline:
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.edgecolor'] = 0.8 ## This is the default
    else:
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.edgecolor'] = 'none'

    return

# Define a custom formatting function
def axis_formatter_talk_freq(freq_value: float, pos) -> str:
    """Format axis for frequency.
    """
    string = ''
    if freq_value >= 1:
        string = f'{freq_value:.0f}'
    else:
        freq_value_base_10 = np.log10(freq_value)
        string = r'$10^{' + str(int(freq_value_base_10)) + '}$'

    return string

def axis_formatter_talk_Rs_Baker(Rs_Baker_value, pos):
    """Format axis for Rs."""
    string = ''
    if Rs_Baker_value >= 1:
        string = f'{Rs_Baker_value:.0f}'
    elif Rs_Baker_value >= 0.1:
        string = f'{Rs_Baker_value:.1f}'
    elif Rs_Baker_value >= 0.01:
        string = f'{Rs_Baker_value:.2f}'
    else:
        log_value = np.log10(Rs_Baker_value)
        string = r'$10^{' + str(int(log_value)) + '}$'
    return string


def plot_twin_y_axis_figure(x1, y1, x2, y2, show_plot=True, title='', x_label='', y1_label='', y2_label='', color1='k', color2=color2_default):
    """
    Plot two dependent variables against an independent variable on a two-axis figure.

    Parameters
    ----------
    x1 : array_like
        Array of the first independent variable values.
    y1 : array_like
        Array of the first dependent variable values.
    x2 : array_like
        Array of the second independent variable values.
    y2 : array_like
        Array of the second dependent variable values.
    show_plot : bool, optional
        If True, the plot is displayed.
    title : str, optional
        The title of the plot.
    x_label : str, optional
        The label of the x axis.
    y1_label : str, optional
        The label of the first y axis.
    y2_label : str, optional
        The label of the second y axis.
    color1 : str, optional
        The color of the first dependent variable.
    color2 : str, optional
        The color of the second dependent variable.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax1 : matplotlib.axes.Axes
        The first axis object.
    ax2 : matplotlib.axes.Axes
        The second axis object.
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    if title != '':
        fig.suptitle(title)

    ax1.set_xlabel(x_label)

    ax1.scatter(x1, y1, s=0.2, color=color1)
    ax1.set_ylabel(y1_label, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2.scatter(x2, y2, s=0.2, color=color2)
    ax2.set_ylabel(y2_label, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Make ax1's background transparent to see ax2 behind it
    
    if show_plot:
        plt.show()
    return fig, ax1, ax2

## Redundant with the broadband file
def Baker_scale(f):
    """
    Calculate the Baker scale for the given frequency.
    
    Parameters
    ----------
    f : float or ndarray
        Frequency [GHz]

    Returns
    -------
    Baker_scale : float or ndarray
        Baker scale [$\\sqrt{\\mu\\Omega\\;cm}$]
    """
    return 1e-4*np.sqrt(mu_0*2*pi*(f*1e9))

## TODO: Update this to save as pdfs
def save_figure_as_svg_png(base_filepath, generate_pdf=True):
    """
    Check if a file exists at the given base filepath, and if not, save the current plt figure to a .svg and .png file at that filepath.

    If a file already exists at the filepath, print a message and do nothing.

    Parameters
    ----------
    base_filepath : str
        The filepath without an extension where the .svg and .png files might be saved.

    Returns
    -------
    None

    Notes
    -----
    - If the file already exists at the filepath, print a message and do nothing.
    - If the file doesn't exist at the filepath, save the current plt figure to a .svg and .png file at that filepath.
    """

    if generate_pdf:
        filepath_svg = base_filepath + '.svg'
        filepath_png = base_filepath + '.png'
        filepath_pdf = base_filepath + '.pdf'

        if os.path.exists(filepath_svg):
            print('File already exists at:', filepath_svg)
        elif os.path.exists(filepath_png):
            print('File already exists at:', filepath_png)
        elif os.path.exists(filepath_pdf):
            print('File already exists at:', filepath_pdf)
        else:
            plt.savefig(filepath_svg, format='svg', transparent=True, bbox_inches='tight', pad_inches=0.1)
            print('Plot saved as .svg at:', filepath_svg)
            plt.savefig(filepath_png, format='png', transparent=True, bbox_inches='tight', pad_inches=0.1, dpi=300)
            print('Plot saved as .png at:', filepath_png)
            plt.savefig(filepath_pdf, format='pdf', transparent=True, bbox_inches='tight', pad_inches=0.1)
            print('Plot saved as .pdf at:', filepath_pdf)
    else:

        filepath_svg = base_filepath + '.svg'
        filepath_png = base_filepath + '.png'

        if os.path.exists(filepath_svg):
            print('File already exists at:', filepath_svg)
        elif os.path.exists(filepath_png):
            print('File already exists at:', filepath_png)
        else:
            # plt.savefig(filepath, format='svg')
            plt.savefig(filepath_svg, format='svg', transparent=True, bbox_inches='tight', pad_inches=0.1)
            print('Plot saved as .svg at:', filepath_svg)
            plt.savefig(filepath_png, format='png', transparent=True, bbox_inches='tight', pad_inches=0.1, dpi=300)
            print('Plot saved as .png at:', filepath_png)


def save_figure_as_svg_png_pdf(base_filepath):
    """
    Check if a file exists at the given base filepath, and if not, save the current plt figure to a .svg and .png file at that filepath.

    If a file already exists at the filepath, print a message and do nothing.

    Parameters
    ----------
    base_filepath : str
        The filepath without an extension where the .svg and .png files might be saved.

    Returns
    -------
    None

    Notes
    -----
    - If the file already exists at the filepath, print a message and do nothing.
    - If the file doesn't exist at the filepath, save the current plt figure to a .svg and .png file at that filepath.
    """

    filepath_svg = base_filepath + '.svg'
    filepath_png = base_filepath + '.png'
    filepath_pdf = base_filepath + '.pdf'

    if os.path.exists(filepath_svg):
        print('File already exists at:', filepath_svg)
    elif os.path.exists(filepath_png):
        print('File already exists at:', filepath_png)
    elif os.path.exists(filepath_pdf):
        print('File already exists at:', filepath_pdf)
    else:
        plt.savefig(filepath_pdf, format='pdf', transparent=True, bbox_inches='tight', pad_inches=0.1)
        print('Plot saved as .pdf at:', filepath_pdf)
        plt.savefig(filepath_svg, format='svg', transparent=True, bbox_inches='tight', pad_inches=0.1)
        print('Plot saved as .svg at:', filepath_svg)
        plt.savefig(filepath_png, format='png', transparent=True, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print('Plot saved as .png at:', filepath_png)
        

## LaTeX string formatting


def convert_vector_to_LaTex_miller(vector: NDArray[np.float64], rounding: int = 0, notation: str = 'symmetric') -> str:
    """
    Convert a vector to a LaTeX representation of the Miller indices.

    Parameters
    ----------
    vector : NDArray[np.float64] of shape (3,)
        The vector to convert to a LaTeX representation of the Miller indices.
    rounding : int, optional
        The number of decimal places to round the vector to.
    notation : str, optional
        The notation to use for the Miller indices.
        - 'symmetric' : Use the symmetry-equivalence notation (eg. <100>)
        - 'specific' : Use the specific notation (eg. [110])

    Returns
    -------
    str
        A string representing the LaTeX representation of the Miller indices.
    """
    miller_string = ''
    # Process q_string
    for j in range(3):
        # Round to the specified number of decimal places
        num = format(abs(vector[j]), f'.{rounding}f')
        if vector[j] < 0:
            miller_string += r'\bar{' + num + r'}'
        else:
            miller_string += num
        
        if j < 2:
            miller_string += r'\,'

    if notation == 'symmetric':
        miller_string = r'$\langle' + miller_string + r'\rangle$'
    elif notation == 'specific':
        miller_string = r'$\left[' + miller_string + r'\right]$'
    else:
        raise ValueError(f'Invalid notation: {notation}')

    return miller_string

def q_direction_LaTeX(vector: NDArray[np.float64], rounding: int = 0) -> str:
    """
    Return the LaTeX notation for q||<Miller>, where <Miller> is the vector's Miller indices.

    Parameters
    ----------
    vector : NDArray[np.float64] of shape (3,)
        The vector to convert to a LaTeX representation of the Miller indices.
    rounding : int, optional
        The number of decimal places to round the vector to.

    Returns
    -------
    str
        String with the LaTeX representation of the q-vector direction in Miller notation.
    """
    q_vector_string = convert_vector_to_LaTex_miller(vector, rounding)
    return r'$\mathbf{q}\parallel$'+q_vector_string

def J_direction_LaTeX(vector: NDArray[np.float64], rounding: int = 0) -> str:
    """
    Return the LaTeX notation for J||<Miller>, where <Miller> is the vector's Miller indices.

    Parameters
    ----------
    vector : NDArray[np.float64] of shape (3,)
        The vector to convert to a LaTeX representation of the Miller indices.
    rounding : int, optional
        The number of decimal places to round the vector to.

    Returns
    -------
    str
        String with the LaTeX representation of the J-vector direction in Miller notation.
    """
    J_vector_string = convert_vector_to_LaTex_miller(vector, rounding)
    return r'$\mathbf{J}\parallel$'+J_vector_string

## String formatting

def format_physical_value_latex(value: float, unit: str, prefix: str = 'auto', precision: str = '.0f') -> str:
    """
    Format a physical value with appropriate SI prefix in LaTeX format.

    Parameters
    ----------
    value : float
        The physical value to format.
    unit : str
        The base unit symbol (e.g., 'Hz', 'm', 's').
    prefix : str, optional
        The SI prefix to use:
        - 'auto': Automatically select a prefix so the displayed value is >= 1
        - Any standard SI prefix: 'Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', '', 
          'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'
    precision : str, optional
        Python format precision string (e.g., '.2g', '.3f').

    Returns
    -------
    str
        LaTeX formatted string of the form '{value} {prefix}{unit}'.
    """
    # SI prefixes and their corresponding powers of 10
    prefixes = {
        'Y': 24,  # yotta
        'Z': 21,  # zetta
        'E': 18,  # exa
        'P': 15,  # peta
        'T': 12,  # tera
        'G': 9,   # giga
        'M': 6,   # mega
        'k': 3,   # kilo
        '': 0,    # (no prefix)
        'm': -3,  # milli
        'μ': -6,  # micro
        'n': -9,  # nano
        'p': -12, # pico
        'f': -15, # femto
        'a': -18, # atto
        'z': -21, # zepto
        'y': -24  # yocto
    }
    
    # For LaTeX, use \\mu instead of μ for the micro symbol
    latex_prefixes = prefixes.copy()
    latex_prefixes['μ'] = latex_prefixes.pop('μ')
    latex_prefix_symbols = {
        'Y': 'Y',
        'Z': 'Z',
        'E': 'E',
        'P': 'P',
        'T': 'T',
        'G': 'G',
        'M': 'M',
        'k': 'k',
        '': '',
        'm': 'm',
        'μ': '\\mu ',
        'n': 'n',
        'p': 'p',
        'f': 'f',
        'a': 'a',
        'z': 'z',
        'y': 'y'
    }
    
    if prefix == 'auto':
        # Find appropriate prefix for auto mode
        if value == 0:
            selected_prefix = ''
        else:
            # Calculate order of magnitude
            magnitude = np.floor(np.log10(abs(value)))
            
            # Find the closest prefix that makes the value between 1 and 1000
            # (i.e., the prefix power should be the highest multiple of 3 
            # that's less than or equal to the magnitude)
            closest_power = 3 * np.floor(magnitude / 3)
            
            # Find the prefix with this power
            selected_prefix = ''
            for prefix_symbol, power in prefixes.items():
                if power == closest_power:
                    selected_prefix = prefix_symbol
                    break
    else:
        # Use specified prefix
        if prefix not in prefixes:
            raise ValueError(f"Invalid prefix: {prefix}. Must be one of {list(prefixes.keys())} or 'auto'")
        selected_prefix = prefix
    
    # Scale the value according to the selected prefix
    scaled_value = value / (10 ** prefixes[selected_prefix])
    
    # Format the value with the specified precision
    formatted_value = f"{{:{precision}}}".format(scaled_value)
    
    # Get the correct LaTeX symbol for the prefix
    latex_prefix = latex_prefix_symbols[selected_prefix]
    
    # Return the LaTeX formatted string
    return r'$' + formatted_value + r'~\mathrm{' + latex_prefix + unit + r'}$'


def format_frequency_latex(frequency: float, prefix: str = 'auto', precision: str = '.0f') -> str:
    """
    Format a frequency value with appropriate SI prefix in LaTeX format.

    Parameters
    ----------
    frequency : float
        The frequency value in Hz.
    prefix : str, optional
        The SI prefix to use:
        - 'auto': Automatically select a prefix so the displayed value is >= 1
        - Any standard SI prefix: 'Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', '', 
          'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'
    precision : str, optional
        Python format precision string (e.g., '.2g', '.3f').

    Returns
    -------
    str
        LaTeX formatted string of the form '{frequency} {prefix}Hz'.
    """
    return format_physical_value_latex(frequency, 'Hz', prefix, precision)


def format_length_latex(length: float, prefix: str = 'auto', precision: str = '.1f') -> str:
    """
    Format a length value with appropriate SI prefix in LaTeX format.

    Parameters
    ----------
    length : float
        The length value in meters.
    prefix : str, optional
        The SI prefix to use:
        - 'auto': Automatically select a prefix so the displayed value is >= 1
        - Any standard SI prefix: 'Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', '', 
          'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'
    precision : str, optional
        Python format precision string (e.g., '.2g', '.3f').

    Returns
    -------
    str
        LaTeX formatted string of the form '{length} {prefix}m'.
    """
    return format_physical_value_latex(length, 'm', prefix, precision)


def format_time_latex(time: float, prefix: str = 'auto', precision: str = '.1f') -> str:
    """
    Format a time value with appropriate SI prefix in LaTeX format.

    Parameters
    ----------
    time : float
        The time value in seconds.
    prefix : str, optional
        The SI prefix to use:
        - 'auto': Automatically select a prefix so the displayed value is >= 1
        - Any standard SI prefix: 'Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', '', 
          'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'
    precision : str, optional
        Python format precision string (e.g., '.2g', '.3f').

    Returns
    -------
    str
        LaTeX formatted string of the form '{time} {prefix}s'.
    """
    return format_physical_value_latex(time, 's', prefix, precision)


def format_resistivity_latex(resistivity: float, prefix: str = 'auto', precision: str = '.0f') -> str:
    """
    Format a resistivity value with appropriate SI prefix in LaTeX format.

    Parameters
    ----------
    resistivity : float
        The resistivity value in Ohm·m.
    prefix : str, optional
        The SI prefix to use:
        - 'auto': Automatically select a prefix so the displayed value is >= 1
        - Any standard SI prefix: 'Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', '', 
          'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'
    precision : str, optional
        Python format precision string (e.g., '.2g', '.3f').

    Returns
    -------
    str
        LaTeX formatted string of the form '{resistivity} {prefix}Ω cm'.
    """
    # Format with the base function, but using a custom unit "Ω·cm"
    # Using \Omega for the ohm symbol in LaTeX and \cdot for the dot product
    return format_physical_value_latex(resistivity*100, '\Omega\\;\\mathrm{cm}', prefix, precision)

