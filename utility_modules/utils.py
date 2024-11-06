"""
General housekeeping utilities.
Functions:
about_win: a toplevel window for the Help>About menu selection.
check_platform - Exit if not Linux, Windows, or macOS.
program_name - Get the program name for file paths and window titles.
valid_path_to - Get correct path to program's files.
set_icon - Set the program icon image file.
save_report_and_img- Save files of result image and its report.
display_report - Place a formatted text string into a specified Frame.
count_sig_fig - Count number of significant figures in a number.
box_centers_very_close - Evaluate nearness of bounding box centers.
box_is_very_close_inarray - List boxes that are very close to another box.
auto_text_contrast - Select text color based on average brightness.
quit_gui -  Error-free and informative exit from the program.
no_objects_found - A simple messagebox when a contour pointset is empty.
get_correction_factor - Calculate the correction factor using a linear equation.
"""
# Copyright (C) 2022-2024 C.S. Echt, under GNU General Public License'

# Standard library imports.
import platform
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Union, List

# Third party imports.
import cv2
import numpy as np
from PIL import ImageTk

# Local application imports.
from utility_modules import manage, constants as const

if const.MY_OS == 'win':
    from ctypes import windll


def about_win() -> None:
    """
    Basic information about the package in scrolling text in a new
    Toplevel window.
    Generally called from a "Help->About" menu.
    Calls manage.arguments() to get the about text.

    Returns:
        None
    """
    aboutwin = tk.Toplevel(bg=const.MASTER_BG)
    aboutwin.title('About Count & Size')
    aboutwin.minsize(width=400, height=200)
    aboutwin.focus_set()
    abouttext = ScrolledText(master=aboutwin,
                             width=62,
                             bg=const.MASTER_BG,  # light gray
                             relief='groove',
                             borderwidth=8,
                             padx=30, pady=10,
                             wrap=tk.WORD,
                             font=const.MENU_FONT,
                             )

    # The text returned from manage.arguments is that used for the --about arg.
    abouttext.insert(index=tk.INSERT,
                     chars=f'{manage.arguments()["about"]}')
    abouttext.grid(sticky=tk.NSEW)


def check_platform() -> None:
    """
    Run check for various platforms to optimize displays.
    Intended to be called at startup.
    """
    if const.MY_OS not in 'win, lin, dar':
        print('Only Windows, Linux, and macOS platforms are supported.\n')
        sys.exit(0)

    # Need to account for Windows scaling in different releases.
    if const.MY_OS == 'win':
        if platform.release() < '10':
            windll.user32.SetProcessDPIAware()
        else:
            windll.shcore.SetProcessDpiAwareness(2)

    # print('To quit, use Esc or Ctrl-Q. From the Terminal, use Ctrl-C.')


def program_name() -> str:
    """
    Returns the script name or, if called from a PyInstaller stand-alone,
    the executable name. Use for setting file paths and naming windows.

    :return: Context-specific name of the main program, as string.
    """
    if getattr(sys, 'frozen', False):  # hasattr(sys, '_MEIPASS'):
        return Path(sys.executable).stem
    return Path(sys.modules['__main__'].__file__).stem


def valid_path_to(input_path: str) -> Path:
    """
    Get correct path to program's directory/file structure
    depending on whether program invocation is a Pyinstaller app or
    the command line. Works with symlinks. Works with absolute paths
    outside of program's folder.
    Allows command line invocation using any path; does not need to be
    from parent directory.

    :param input_path: Program's local dir/file name, as string.
    :return: Absolute path as pathlib Path object_type.
    """
    # Note that Path(Path(__file__).parent is the utility_modules folder.
    # Modified from: https://stackoverflow.com/questions/7674790/
    #    bundling-data-files-with-pyinstaller-onefile and PyInstaller manual.
    if getattr(sys, 'frozen', False):  # hasattr(sys, '_MEIPASS'):
        base_path = getattr(sys, '_MEIPASS', Path(Path(__file__).resolve()).parent)
        valid_path = Path(base_path) / input_path
    else:
        valid_path = Path(f'{input_path}').resolve()

    return valid_path


def set_icon(window: Union[tk.Toplevel, tk.Tk]) -> None:
    """
    Set the program icon image file.  If the icon cannot be displayed,
    print a message to the console.

    Args:  window: The main tk.Tk() window running the mainloop or a
        tk.Toplevel window.
    """
    # The custom app icon is expected to be in the program's images folder.
    try:
        icon = tk.PhotoImage(file=valid_path_to('images/oystersize_icon256.png'))
        window.wm_iconphoto(True, icon)
    except tk.TclError as err:
        print('Cannot display program icon, so it will be blank or the tk default.\n'
              f'tk error message: {err}')
    except FileNotFoundError as fnf:
        print(f'Cannot find program icon file: {fnf}.\n'
              'The program will run without a custom icon image.')


def save_report_and_img(path2folder: str,
                        img2save: Union[np.ndarray, ImageTk.PhotoImage],
                        txt2save: str,
                        caller: str,
                        ) -> None:
    """
    Write to file the current report of calculated image processing
    values. Save current result image or selected displayed image.

    Args:
        path2folder: The input image file path, as string.
        img2save: The current resulting image array; can be a np.ndarray
            from cv2 or an ImageTk.PhotoImage from tkinter/PIL
        txt2save: The current image processing report.
        caller: Descriptive name of the calling script, function or
                widget to insert in the file name, e.g. 'report',
                'contrast', etc.
    Returns: None
    """
    time_now = datetime.now().strftime(const.TIME_STAMP_FORMAT)
    time2print = datetime.now().strftime(const.TIME_PRINT_FORMAT)

    # For JPEG file format the supported parameter is cv2.IMWRITE_JPEG_QUALITY
    # with a possible value between 0 and 100, the default value being 95.
    # The higher value produces a better quality image file.
    #
    # For PNG file format the supported imwrite parameter is
    # cv2.IMWRITE_PNG_COMPRESSION with a possible value between 0 and 9,
    # the default being 3. The higher value does high compression of the
    # image resulting in a smaller file object_size but a longer compression time.

    img_ext = Path(path2folder).suffix
    img_name = Path(path2folder).stem
    img_folder = Path(path2folder).parent
    saved_report_name = f'{img_name}_{caller}_Report.txt'
    report_file_path = Path(img_folder, saved_report_name)
    saved_img_name = f'{img_name}_{caller}_{time_now}{img_ext}'
    saved_img_path = f'{img_folder}/{saved_img_name}'

    if manage.arguments()['terminal']:
        data2save = (f'\n\nTime saved: {time2print}\n'
                     f'Saved image file: {saved_img_name}\n'
                     f'Saved report file: {saved_report_name}\n'
                     f'{txt2save}')
    else:
        data2save = (f'\n\nTime saved: {time2print}\n'
                     f'Saved image file: {saved_img_name}\n'
                     f'{txt2save}')

    # Use this Path function for saving individual report files:
    #   Path(f'{img_stem}_{caller}_settings{time_now}.txt').write_text(data2save)
    # Use this for appending multiple reports to single file:
    with open(report_file_path, mode='a', encoding='utf-8') as fp:
        fp.write(data2save)

    # Contour images are np.ndarray direct from cv2 functions, while
    #   other images are those displayed as ImageTk.PhotoImage.
    if isinstance(img2save, np.ndarray):
        cv2.imwrite(filename=saved_img_path, img=img2save)
    elif isinstance(img2save, ImageTk.PhotoImage):
        # Need to get the ImageTK image into a format that can be saved to file.
        # source: https://stackoverflow.com/questions/45440746/
        #   how-to-save-pil-imagetk-photoimage-as-jpg
        imgpil = ImageTk.getimage(img2save)

        # source: https://stackoverflow.com/questions/48248405/
        #   cannot-write-mode-rgba-as-jpeg
        if imgpil.mode in ("RGBA", "P"):
            imgpil = imgpil.convert("RGB")

        imgpil.save(saved_img_path)
    else:
        print('The specified image needs to be a np.ndarray or ImageTk.PhotoImage ')

    if manage.arguments()['terminal']:
        print(f'Result image and its report were saved to files.'
              f'{data2save}')


def display_report(frame: tk.Frame, report: str) -> None:
    """
    Places a formatted text string into the specified Frame; allows for
    real-time updates of text and proper alignment of text in the Frame.

    Args:
        frame: The tk.Frame() in which to place the *report* text.
        report: Text string of values, data, etc. to report.

    Returns: None
    """

    max_line = len(max(report.splitlines(), key=len))

    reporttxt = ScrolledText(master=frame,
                             font=const.REPORT_FONT,
                             bg=const.DARK_BG,
                             fg=const.COLORS_TK['yellow'],  # Matches slider labels.
                             width=max_line,
                             height=report.count('\n'),
                             relief='flat',
                             padx=8, pady=8,
                             wrap=tk.NONE,
                             )

    # Replace prior Text with current text;
    #  hide cursor in Text;
    #  always show start of last line with window resize;
    #  (re-)grid in-place.
    reporttxt.delete(index1='1.0', index2=tk.END)
    reporttxt.insert(index=tk.INSERT, chars=report)
    # Indent helps center text in the Frame.
    reporttxt.tag_configure(tagName='leftmargin', lmargin1=20)
    reporttxt.tag_add('leftmargin', '1.0', tk.END)
    reporttxt.configure(state=tk.DISABLED)
    reporttxt.see(index="end-1c linestart")
    reporttxt.grid(column=0, row=0, columnspan=2, sticky=tk.NSEW)


def count_sig_fig(entry_number: Union[int, float, str]) -> int:
    """
    Determine the number of significant figures in a number.
    Be sure to verify that *entry_number* is a real number prior to using
    it as a parameter.
    The sigfig length value returned here can be used as the 'precision'
    parameter value in to_p.to_precision() statements.

    Args:
        entry_number: Any numerical representation, as string or digits.

    Returns: Integer count of significant figures in *entry_number*.
    """

    # See: https://en.wikipedia.org/wiki/Significant_figures#Significant_figures_rules_explained
    # Grab only numeric characters from *entry_number*
    number_str = str(entry_number).lower()
    sigfig_str = ''.join(_c for _c in number_str if _c.isnumeric())

    # If in scientific notation, remove the trailing exponent value.
    #  The exponent and exp_len statements allow any object_size of e power.
    #  Determine only absolute value of exponent to get its string length.
    #  Accounts for 'e0x' and 'e-0x' expressions.
    if 'e' in number_str:
        abs_exp = abs(int(number_str.split('e')[-1]))
        sigfig_str = sigfig_str[:-len(str(abs_exp))]

    # Finally, remove leading zeros, which are not significant, and
    #  return the total count of significant figures.
    return len(sigfig_str.lstrip('0'))


def box_centers_very_close(box_a: list, box_b: list) -> bool:
    """
    Evaluate nearness of standard and oyster bounding boxes that are
    center-oriented, as in YOLO format: x-ctr, y-ctr, width, height.
    Boxes are considered very close if the centers are within 10% of
    the additive width and height of both boxes.
    Needed because sometimes the YOLO model will detect an oyster
    as both a standard (disk) and as an oyster. This method is used
    to help evaluate whether to remove the standard from the object_name
    list of objects used for calculation and annotation.

    Args:
        box_a: A numpy array element for a bounding box, in [x y w h] format.
        box_b: A numpy array element for another bounding box, in [x y w h] format.
    Returns: True if the boxes are close, False if not.
    """
    # https://gamedev.stackexchange.com/questions/586/
    #   what-is-the-fastest-way-to-work-out-2d-bounding-box-intersection
    # Closeness threshold is 10% of the sum of the widths and heights.
    return ((abs(box_a[0] - box_b[0]) < (box_a[2] + box_b[2]) * 0.1) and
                (abs(box_a[1] - box_b[1]) < (box_a[3] + box_b[3]) * 0.1))


def box_is_very_close_inarray(box: List[np.ndarray],
                              arr: np.ndarray) -> List[np.ndarray]:
    """
    Return a list of boxes from the *box* numpy array that are very clos
    to any element in the *arr* array of bounding boxes nearness. When
    used with the box_centers_very_close() method and called from a
    statement that evaluates the negative of being very close, this can
    be an effective filter to obtain well-separated objects of different
    YOLO classes.

    Args: box: A single bounding box, in [[x y w h]] format.
          arr: A numpy array of bounding boxes, each in [x y w h] format.
    Returns: A list of strongly overlapping numpy array bounding boxes,
             in [x y w h] format.
    """
    return [ary_box for ary_box in arr if box_centers_very_close(box, ary_box)]


def auto_text_contrast(box_area: np.ndarray, center_pct: float) -> tuple:
    """
    Select text contrast, black or white, based on average brightness of
    the specified central area *center_pct*, as a percentage of an
    object's bounding box *box_area*.

    Args:
        box_area: A numpy array of an object's bounding box area.
        center_pct: A decimal proportion of the box's central area used
                    to calculate an average luminance value.
    Returns:
        A tuple of the BGR color values selected for contrasting text
        annotation of black or white.
    """

    # Center given percentage of the box for average color determination.
    center_factor = (1 - center_pct) / 2
    top_row = int(box_area.shape[0] * center_factor)
    bottom_row = int(box_area.shape[0] * (1 - center_factor))
    left_column = int(box_area.shape[1] * center_factor)
    right_column = int(box_area.shape[1] * (1 - center_factor))
    center_area = box_area[top_row:bottom_row, left_column:right_column, :]

    # Calculate perceived brightness of the center area for contrast determination.
    # https://www.nbdtech.com /Blog/archive/2008/04/27/
    #   Calculating-the-Perceived-Brightness-of-a-Color.aspx
    # Cutoff of perceived brightness, -pb, in range(128-145) to switch from
    #   black to white foreground will give acceptable visual contrast when
    #   background below that PB. 128 is the midpoint of 256, the max PB.
    # Range of 128-145 will give acceptable results, says author @NirDobovizki.
    _B, _G, _R = np.mean(center_area, axis=0).mean(axis=0)
    _pb = ((.068 * _B ** 2) + (.691 * _G ** 2) + (.241 * _R ** 2)) ** 0.5
    if _pb > 145:
        return const.COLORS_CV['black']
    return const.COLORS_CV['white']


def quit_gui(mainloop: tk.Tk, confirm=True) -> None:
    """Safe and informative exit from the program."""

    def _do_quit():
        print('...User has quit the program')
        try:
            mainloop.update()
            mainloop.destroy()
            sys.exit(0)  # just in case the mainloop doesn't close
        except Exception as unk:
            print('An unknown error occurred:', unk)
            sys.exit(0)

    if confirm:
        try:
            really_quit = messagebox.askyesno(
                parent=mainloop.focus_get(),
                title="Confirm Exit",
                detail='Are you sure you want to quit?')
            if really_quit:
                _do_quit()
        except (tk.TclError, KeyError) as tkerr:
            print(f'An error occurred during exit: {tkerr}')
            _do_quit()
    else:
        _do_quit()

def no_objects_found_msg(caller: str) -> None:
    """
    Pop-up info when segments not found or their sizes out of range.

    Args:
        caller: The calling statement context; e.g. 'predicted_boxes',
            'std_objects', 'oyster_objects', 'interior_std', 'interior_oyster'.
    """
    messages = {
        'predicted_boxes': '\nNo objects were recognized.\n'
                           'Try changing the confidence threshold.\n'
                           'Or try a different image.\n\n',
        'std_objects': '\nNo size standards were recognized.\n'
                       'Try changing the confidence threshold.\n'
                       'Or include a disk of known diameter in image\n\n',
        'interior_std': '\nNo non-border size standards were recognized.\n'
                        'Try changing the confidence threshold.\n'
                        'Or try a different image with standard farther from edge.\n',
        'interior_oyster': '\nNo non-border oysters were recognized.\n'
                           'Try changing the confidence threshold.\nO'
                           'Or try a different image with oysters farther from edges.\n',
        'oyster_objects': '\nNo oysters were recognized.\n'
                          'Try changing the confidence threshold.\n'
                          'Or try a different image.\n\n'
    }
    messagebox.showinfo(detail=messages.get(caller, 'No objects found: reason unknown.'))


def get_correction_factor(box_ratio_mean: float) -> float:
    """
    Calculate the correction factor using a linear equation derived from
    the correction_factors dictionary.

    Args:
        box_ratio_mean: The mean ratio of the bounding box dimensions.

    Returns:
        The correction factor as a float.
    """
    # From GitHub Copilot inline chat:
    # To derive a linear equation for the correction factors, we can use the formula for
    # a linear equation \( y = mx + b \), where \( m \) is the slope and \( b \) is the y-intercept.
    #
    # We can calculate the slope \( m \) and the  y-intercept \( b \) using two points from
    # the correction_factors dictionary in constants.py.
    # Let's use the points (1.15, 1.005) and (1.36, 1.11) for (box_ratio, correction factor).
    # 1. Calculate the slope \( m \):
    # \[ m = \frac{y_2 - y_1}{x_2 - x_1} = \frac{1.11 - 1.005}{1.36 - 1.15} = \frac{0.105}{0.21} = 0.5 \]
    # 2. Calculate the y-intercept \( b \):
    # \[ b = y_1 - mx_1 = 1.005 - (0.5 \times 1.15) = 1.005 - 0.575 = 0.43 \]
    # So, the linear equation is:
    # \[ y = 0.5x + 0.43 \]

    if box_ratio_mean <= const.BOX_RATIO_THRESHOLD:
        return 1.0

    return 0.5 * box_ratio_mean + 0.43
